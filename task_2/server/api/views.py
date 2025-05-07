import cv2
from PIL import Image
import numpy as np
from insightface.app import FaceAnalysis
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser

from .serializers import FaceVerifySerializer


BUFFALO_L = FaceAnalysis(name="buffalo_l")
BUFFALO_L.prepare(ctx_id=0, det_size=(256, 256), det_thresh=0.2)

BUFFALO_S = FaceAnalysis(name="buffalo_s")
BUFFALO_S.prepare(ctx_id=0, det_size=(256, 256), det_thresh=0.2)


class FaceVerifyView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):

        serializer = FaceVerifySerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        img1 = serializer.validated_data['img1']
        img2 = serializer.validated_data['img2']
        model_name = serializer.validated_data['model']

        img1 = np.array(Image.open(img1))
        img2 = np.array(Image.open(img2))

        if model_name == 'buffalo_s':
            model = BUFFALO_S
        elif model_name == 'buffalo_l':
            model = BUFFALO_L
        else:
            raise ValueError(f"Invalid model name `{model_name}`")

        faces1 = model.get(img1)
        faces2 = model.get(img2)

        if not faces1 or not faces2:
            return Response({'error': 'Face not detected in one or both images.'}, status=status.HTTP_400_BAD_REQUEST)

        emb1 = faces1[0].embedding
        emb2 = faces2[0].embedding

        sim = float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))

        return Response({'similarity': sim}, status=status.HTTP_200_OK)