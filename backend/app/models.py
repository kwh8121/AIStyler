from datetime import datetime
from zoneinfo import ZoneInfo
from pydantic import BaseModel, ConfigDict, field_serializer

class CustomModel(BaseModel):
    """
    프로젝트의 모든 Pydantic 스키마가 상속받는 공통 기본 모델.
    API 데이터 정책을 중앙에서 관리합니다.
    """
    model_config = ConfigDict(
        # True일 경우, 필드 별칭(alias)으로도 값을 할당할 수 있습니다.
        populate_by_name=True,
        
        # SQLAlchemy 모델 객체를 Pydantic 스키마로 변환 가능하게 합니다.
        from_attributes=True,

        # 모든 필드가 필수 필드가 아니라면, 모델 생성 시 필요한 필드만 지정할 수 있습니다.
        extra="forbid",
    )
    
    @field_serializer('*', check_fields=False)
    def serialize_datetime(self, value, _info):
        """datetime 객체를 서울 시간대 기준으로 특정 포맷의 문자열로 변환합니다."""
        if isinstance(value, datetime):
            # 만약 시간대 정보가 없는 naive datetime이라면, 서울 시간대로 간주합니다.
            seoul_tz = ZoneInfo("Asia/Seoul")
            if value.tzinfo is None:
                value = value.replace(tzinfo=seoul_tz)
            else:
                # 시간대 정보가 이미 있다면, 서울 시간대로 변환합니다.
                value = value.astimezone(seoul_tz)
            return value.strftime("%Y-%m-%d %H:%M:%S")  # 프론트엔드와 협의된 명확한 포맷 사용
        return value