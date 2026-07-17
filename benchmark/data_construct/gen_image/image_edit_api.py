import importlib
import os
from dataclasses import dataclass
from typing import Any


CLIENT_CLASS_ENV = "IMAGE_EDIT_API_CLIENT"
ACCESS_KEY_ENV = "IMAGE_EDIT_API_ACCESS_KEY"
SECRET_KEY_ENV = "IMAGE_EDIT_API_SECRET_KEY"
SUBMIT_URL_ENV = "IMAGE_EDIT_API_SUBMIT_URL"
QUERY_URL_TEMPLATE_ENV = "IMAGE_EDIT_API_QUERY_URL_TEMPLATE"


@dataclass
class ImageEditBackend:
    client: Any
    submit_url: str
    query_url_template: str

    def submit(self, header: dict, payload: str) -> dict:
        return self.client.post(self.submit_url, header, payload)

    def query(self, task_id: str) -> dict:
        return self.client.get(self.query_url_template.format(task_id=task_id))


def add_image_edit_api_args(parser) -> None:
    parser.add_argument(
        "--image-edit-api-client",
        default=os.environ.get(CLIENT_CLASS_ENV),
        help=f"Import path for a signed HTTP API client class. Can also use {CLIENT_CLASS_ENV}.",
    )
    parser.add_argument(
        "--image-edit-api-access-key",
        default=os.environ.get(ACCESS_KEY_ENV),
        help=f"Image editing API access key. Can also use {ACCESS_KEY_ENV}.",
    )
    parser.add_argument(
        "--image-edit-api-secret-key",
        default=os.environ.get(SECRET_KEY_ENV),
        help=f"Image editing API secret key. Can also use {SECRET_KEY_ENV}.",
    )
    parser.add_argument(
        "--image-edit-api-submit-url",
        default=os.environ.get(SUBMIT_URL_ENV),
        help=f"Image editing task submit URL. Can also use {SUBMIT_URL_ENV}.",
    )
    parser.add_argument(
        "--image-edit-api-query-url-template",
        default=os.environ.get(QUERY_URL_TEMPLATE_ENV),
        help=(
            f"Image editing task query URL template with {{task_id}}. "
            f"Can also use {QUERY_URL_TEMPLATE_ENV}."
        ),
    )


def build_image_edit_backend(args, parser=None) -> ImageEditBackend:
    required = {
        "--image-edit-api-client": args.image_edit_api_client,
        "--image-edit-api-access-key": args.image_edit_api_access_key,
        "--image-edit-api-secret-key": args.image_edit_api_secret_key,
        "--image-edit-api-submit-url": args.image_edit_api_submit_url,
        "--image-edit-api-query-url-template": args.image_edit_api_query_url_template,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        message = "Missing required image editing API config: " + ", ".join(missing)
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    if "{task_id}" not in args.image_edit_api_query_url_template:
        message = "--image-edit-api-query-url-template must include the {task_id} placeholder"
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    client_class = _load_client_class(args.image_edit_api_client)
    client = client_class(args.image_edit_api_access_key, args.image_edit_api_secret_key)
    return ImageEditBackend(
        client=client,
        submit_url=args.image_edit_api_submit_url,
        query_url_template=args.image_edit_api_query_url_template,
    )


def _load_client_class(import_path: str):
    if ":" in import_path:
        module_name, class_name = import_path.split(":", 1)
    else:
        module_name, _, class_name = import_path.rpartition(".")

    if not module_name or not class_name:
        raise ValueError(
            "Image editing API client must be an import path like "
            "package.module:ClientClass or package.module.ClientClass"
        )

    module = importlib.import_module(module_name)
    return getattr(module, class_name)
