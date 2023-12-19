from logging import Logger
from fastapi import APIRouter, Request, Depends
from fastapi.responses import JSONResponse

from ASC3.mil_model.model import MILPredictor
from ASC3.mil_model.data_model import SampleId, MILRequest


mil_router = APIRouter()


def get_logger(request: Request) -> Logger:
    return request.app.state.logger


def get_predictor(request: Request) -> MILPredictor:
    return request.app.state.mil_predictor


@mil_router.post("/predict_from_file")
def predict_from_file(
    query: SampleId,
    mil_predictor: MILPredictor = Depends(get_predictor),
    logger: Logger = Depends(get_logger),
) -> JSONResponse:
    """
    주어진 샘플을 기반으로 MIL (Multiple Instance Learning)을 사용하여 결과를 예측

    Note:
        파일로부터 PatientData을 생성할 때, PatientData.snv_data.header의
        값을 API payload 포맷에 맞추기위해 해더명도 함께 변경함

    Args:
        sample_id (str): 예측에 사용할 샘플의 식별자
        request (Request): FastAPI의 요청 객체

    Returns:
        JSONResponse: 예측된 Bag 확률과 원인변이 스코어를 담고 있는 JSON 응답
    """
    sample_id = query.sample_id
    logger.info("Passed sample id %s" % sample_id)

    patient_data = mil_predictor.build_data_from_file(sample_id)
    bag_label, variant2score = mil_predictor.predict(patient_data)

    return JSONResponse(
        content={"patient_probability": bag_label, "variant_probability": variant2score}
    )


@mil_router.post("/predict")
def predict(
    query: MILRequest,
    mil_predictor: MILPredictor = Depends(get_predictor),
    logger: Logger = Depends(get_logger),
) -> JSONResponse:
    """특징값을 POST 요청을 받아서 MIL(Multiple Instance Learning) 모델을 사용하여 예측

    Args:
        query (MILRequest): POST 요청에서 받은 데이터를 나타내는 MILRequest 객체.
        request (Request): FastAPI Request 객체.

    Returns:
        JSONResponse: 예측 결과를 JSON 형식으로 반환
            반환되는 JSON은 다음과 같은 형식을 따릅니다:
            {
                "bag_prob": 각 환자에 대한 확률,
                "variants": 각 변형(데이터 포인트)에 대한 점수
            }
    """
    sample_id = query.sample_id
    logger.info("Passed sample id %s" % sample_id)

    patient_data = mil_predictor.convert_query_to_patient_data(query)
    bag_prob, variant2score = mil_predictor.predict(patient_data)

    return JSONResponse(
        content={"patient_probability": bag_prob, "variant_probability": variant2score}
    )
