from fastapi import FastAPI
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError


async def validation_exception_handler(request: Request, exception) -> JSONResponse:
    logger = request.app.state.logger
    errors = exception.errors()

    msg = []
    for error in errors:
        error_type = error["type"]
        error_loc = error["loc"]
        error_msg = error["msg"]
        msg.append(f"{error_type} at {error_loc}, {error_msg}")
    logger.error("//".join(msg))

    return JSONResponse(errors, status_code=422)


def add_exception_handlers(app: FastAPI) -> None:
    """여러 예외처리의 핸들링 추가

    Args:
        app (FastAPI): 예외를 처리할 FastAPI 객체

    Note:
        https://github.com/tiangolo/fastapi/issues/3388
    """
    app.add_exception_handler(RequestValidationError, handler=validation_exception_handler)
