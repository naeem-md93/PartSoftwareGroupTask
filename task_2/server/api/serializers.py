from rest_framework import serializers


class FaceVerifySerializer(serializers.Serializer):
    img1 = serializers.FileField()
    img2 = serializers.FileField()
    model = serializers.ChoiceField(choices=["buffalo_l", "buffalo_s"])