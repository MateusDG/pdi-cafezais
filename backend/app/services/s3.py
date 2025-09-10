from typing import Annotated
import boto3
from botocore.client import Config
from fastapi import Depends
from ..core.config import settings

class S3Client:
    def __init__(self):
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.minio_endpoint,
            aws_access_key_id=settings.minio_access_key,
            aws_secret_access_key=settings.minio_secret_key,
            config=Config(signature_version="s3v4")
        )
        self._bucket = settings.minio_bucket_images

    def put_object(self, key: str, data: bytes, content_type: str | None = None):
        extra = {}
        if content_type:
            extra["ContentType"] = content_type
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data, **extra)

    @staticmethod
    def depends() -> Annotated["S3Client", Depends]:
        return S3Client()
