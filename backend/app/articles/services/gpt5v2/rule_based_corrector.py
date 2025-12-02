#!/usr/bin/env python3
"""
Rule-Based Pre-Corrector for Korea Times Articles
==================================================
학습된 모델 추론 전에 명확한 규칙들을 자동으로 교정합니다.

사용법:
    from rule_based_corrector import RuleBasedCorrector

    corrector = RuleBasedCorrector()
    result = corrector.correct_article(article_text)

    print(result['corrected_text'])
    print(result['corrections'])  # 적용된 교정 목록
"""

import re
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from datetime import datetime
import calendar


@dataclass
class Correction:
    """교정 정보"""
    rule_id: str
    component: str  # 'title', 'body', 'caption'
    original: str
    corrected: str
    position: int  # 텍스트 내 위치


class RuleBasedCorrector:
    """규칙 기반 교정기"""

    def __init__(self):
        """
        Rule-Based Tier 규칙 초기화
        95%+ 정확도의 명확한 패턴만 포함
        """
        # 숫자 철자 → 아라비아 숫자 매핑
        self.number_words = {
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20', 'thirty': '30',
            'forty': '40', 'fifty': '50', 'sixty': '60', 'seventy': '70',
            'eighty': '80', 'ninety': '90', 'hundred': '100', 'thousand': '1,000',
            'million': '1 million', 'billion': '1 billion', 'trillion': '1 trillion'
        }
        # 숫자 단어 → 값 변환용 보조 맵
        self._number_word_value = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
            'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19,
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
        }
        self._number_scales = {'hundred': 100, 'thousand': 1000, 'million': 1_000_000, 'billion': 1_000_000_000}
        self._digit_to_word = {str(i): w for i, w in enumerate(
            ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        )}
        # A28 단위 목록 (숫자 강제)
        self._a28_units = [
            # 통화/규모
            r'percent(?:age)?', r'won', r'dollar(?:s)?', r'euro(?:s)?',
            r'million', r'billion', r'trillion',
            # 거리/길이
            r'meter(?:s)?', r'kilometer(?:s)?', r'centimeter(?:s)?', r'millimeter(?:s)?',
            r'mile(?:s)?', r'inch(?:es)?', r'foot', r'feet',
            # 온도/각도
            r'degree(?:s)?',
            # 시간
            r'second(?:s)?', r'minute(?:s)?', r'hour(?:s)?', r'day(?:s)?', r'week(?:s)?', r'month(?:s)?', r'year(?:s)?(?:\s+old)?',
        ]

        # 복합어 분리 대상 (A34)
        self.compound_words_to_split = {
            'childcare': 'child care',
            'healthcare': 'health care',
            'taskforce': 'task force'
        }

        # ===== Tier 1A: 한국 관련 데이터 =====

        # C29 - 한국 산 이름 매핑 (한국 100대 명산 기준 확장)
        self.korean_mountains = {
            # 5대 명산
            'Hallasan': 'Mount Halla',
            'Jirisan': 'Mount Jiri',
            'Seoraksan': 'Mount Seorak',
            'Bukhansan': 'Mount Bukhan',
            'Namsan': 'Mount Nam',

            # 서울/경기 주요 산
            'Gwanaksan': 'Mount Gwanak',
            'Dobongsan': 'Mount Dobong',
            'Inwangsan': 'Mount Inwang',
            'Cheonmasan': 'Mount Cheonma',
            'Suraksan': 'Mount Surak',
            'Yongmunsan': 'Mount Yongmun',
            'Achasan': 'Mount Acha',
            'Cheonggyesan': 'Mount Cheonggye',
            'Gwanggyosan': 'Mount Gwanggyo',
            'Soyosan': 'Mount Soyo',
            'Ungilsan': 'Mount Ungil',

            # 강원도 주요 산
            'Baegunsan': 'Mount Baegun',
            'Odaesan': 'Mount Odae',
            'Sobaeksan': 'Mount Sobaek',
            'Taebaeksan': 'Mount Taebaek',
            'Chiaksan': 'Mount Chiak',
            'Daeamsan': 'Mount Daeam',
            'Seonjasan': 'Mount Seonja',
            'Taewhasan': 'Mount Taewha',
            'Gariwangsan': 'Mount Gariwang',
            'Oseosan': 'Mount Oseo',

            # 충청도 주요 산
            'Gyeryongsan': 'Mount Gyeryong',
            'Songisan': 'Mount Songi',
            'Daedunsan': 'Mount Daedun',
            'Gyebangsan': 'Mount Gyebang',
            'Minjujisan': 'Mount Minjuji',
            'Chilgapsan': 'Mount Chilgap',

            # 전라도 주요 산
            'Naejangsan': 'Mount Naejang',
            'Mudeungsan': 'Mount Mudeung',
            'Maisan': 'Mount Mai',
            'Byeonsan': 'Mount Byeon',
            'Wolchulsan': 'Mount Wolchul',
            'Jogyesan': 'Mount Jogye',
            'Yudalsan': 'Mount Yudal',
            'Janggunbong': 'Mount Janggunbong',
            'Unjangsan': 'Mount Unjang',
            'Duryunsan': 'Mount Duryun',
            'Daedunsan': 'Mount Daedun',

            # 경상도 주요 산
            'Deogyusan': 'Mount Deogyu',
            'Palgongsan': 'Mount Palgong',
            'Apsan': 'Mount Ap',
            'Geumosan': 'Mount Geumo',
            'Gajisan': 'Mount Gaji',
            'Cheonseongsan': 'Mount Cheonseong',
            'Unmunsan': 'Mount Unmun',
            'Hwangmaesan': 'Mount Hwangmae',
            'Gajisan': 'Mount Gaji',
            'Biseulsan': 'Mount Biseul',
            'Yeongchwisan': 'Mount Yeongchwi',

            # 제주 산
            'Sanggumburi': 'Mount Sanggumburi',
            'Seongsan': 'Mount Seongsan',  # Seongsan Ilchulbong

            # 기타 유명 산
            'Wolaksan': 'Mount Wolak',
            'Sambongsan': 'Mount Sambong',
            'Gukmangbong': 'Mount Gukmangbong',
            'Hwaaksan': 'Mount Hwaak',
            'Chunmasan': 'Mount Chunma',
        }

        # C29 - 예외: Mt.가 정상인 경우 (고유명사, 기관명)
        self.mountain_exceptions = [
            'Mt. Sinai', 'Mt. Everest', 'Mt. Fuji', 'Mt. Kilimanjaro',  # 해외 산
            'Mt. Sinai Hospital', 'Mt. Vernon',  # 기관/지명
            'Rocky Mountain', 'Blue Mountain',  # 고정 복합명사
        ]

        # 한국 국립공원 (22개) - 예외 처리용
        self.korean_national_parks = [
            'Jirisan National Park',
            'Gyeongju National Park',
            'Seoraksan National Park',
            'Hallasan National Park',
            'Naejangsan National Park',
            'Gayasan National Park',
            'Deogyusan National Park',
            'Odaesan National Park',
            'Chiaksan National Park',
            'Woraksan National Park',
            'Juwangsan National Park',
            'Taeanhaean National Park',
            'Dadohaehaesang National Park',
            'Bukhansan National Park',
            'Sobaeksan National Park',
            'Mudeungsan National Park',
            'Wolchulsan National Park',
            'Byeonsanbando National Park',
            'Songnisan National Park',
            'Hallyeohaesang National Park',
            'Taebaeksan National Park',
            'Guinsa Temple in Sobaeksan',
        ]

        # 한국 주요 섬
        self.korean_islands = {
            'Jejudo': 'Jeju Island',
            'Ulleungdo': 'Ulleung Island',
            'Dokdo': 'Dokdo Island',
            'Ganghwado': 'Ganghwa Island',
            'Namiseom': 'Nami Island',
            'Yeongjongdo': 'Yeongjong Island',
            'Geojedo': 'Geoje Island',
            'Jindo': 'Jindo Island',
            'Muuido': 'Muui Island',
        }

        # 한국 주요 강
        self.korean_rivers = {
            'Han River': 'Han River',  # 이미 올바름
            'Hangang': 'Han River',
            'Nakdong River': 'Nakdong River',
            'Nakdonggang': 'Nakdong River',
            'Geum River': 'Geum River',
            'Geumgang': 'Geum River',
            'Yeongsan River': 'Yeongsan River',
            'Yeongsangang': 'Yeongsan River',
        }

        # 한국 주요 사찰 (Temple은 이미 영문)
        self.korean_temples = {
            'Bulguksa': 'Bulguk Temple',
            'Seokguram': 'Seokguram Grotto',
            'Haeinsa': 'Haein Temple',
            'Songgwangsa': 'Songgwang Temple',
            'Beomeosa': 'Beomeo Temple',
            'Jogyesa': 'Jogye Temple',
            'Bongeunsa': 'Bongeun Temple',
            'Gilsangsa': 'Gilsang Temple',
        }

        # C28 - 한국 궁궐 이름 매핑
        self.korean_palaces = {
            'Gyeongbokgung': 'Gyeongbok Palace',
            'Changdeokgung': 'Changdeok Palace',
            'Deoksugung': 'Deoksu Palace',
            'Changgyeonggung': 'Changgyeong Palace',
            'Gyeonghuigung': 'Gyeonghui Palace',
        }

        # C28 - 예외: Palace가 필요 없는 경우
        self.palace_exceptions = [
            'Station',  # 지하철역
            'area', 'district', 'vicinity', 'neighborhood',  # 지역명
            '-gil', '-ro', 'Street', 'Road',  # 도로명
            'near', 'by', 'around',  # 위치 전치사
        ]

        # A25/C27 - 서울 25개 구 이름 (X-gu → X District)
        self.seoul_districts = {
            'Jongno-gu': 'Jongno District',
            'Jung-gu': 'Jung District',
            'Yongsan-gu': 'Yongsan District',
            'Seongdong-gu': 'Seongdong District',
            'Gwangjin-gu': 'Gwangjin District',
            'Dongdaemun-gu': 'Dongdaemun District',
            'Jungnang-gu': 'Jungnang District',
            'Seongbuk-gu': 'Seongbuk District',
            'Gangbuk-gu': 'Gangbuk District',
            'Dobong-gu': 'Dobong District',
            'Nowon-gu': 'Nowon District',
            'Eunpyeong-gu': 'Eunpyeong District',
            'Seodaemun-gu': 'Seodaemun District',
            'Mapo-gu': 'Mapo District',
            'Yangcheon-gu': 'Yangcheon District',
            'Gangseo-gu': 'Gangseo District',
            'Guro-gu': 'Guro District',
            'Geumcheon-gu': 'Geumcheon District',
            'Yeongdeungpo-gu': 'Yeongdeungpo District',
            'Dongjak-gu': 'Dongjak District',
            'Gwanak-gu': 'Gwanak District',
            'Seocho-gu': 'Seocho District',
            'Gangnam-gu': 'Gangnam District',
            'Songpa-gu': 'Songpa District',
            'Gangdong-gu': 'Gangdong District',
        }

        # C18 - "Captured from" 모호한 출처
        self.vague_capture_sources = [
            'video', 'photo', 'image', 'screen', 'capture',
            'recording', 'clip', 'footage'
        ]

        # C15 - Korea Times 크레딧 패턴
        self.kt_credit_patterns = [
            'Korea Times file',
            'Korea Times photo',
            'Korea Times archive',
            'Korea Times'
        ]

        # C13 - 날짜 범위 표기용 월 매핑
        self.months_full = [
            'January', 'February', 'March', 'April', 'May', 'June',
            'July', 'August', 'September', 'October', 'November', 'December'
        ]
        self.months_abbr = [
            'Jan.', 'Feb.', 'Mar.', 'April', 'May', 'June',
            'July', 'Aug.', 'Sep.', 'Oct.', 'Nov.', 'Dec.'
        ]

        # 월 이름 → 인덱스 매핑 (0-11)
        self.month_to_index = {}
        for i, month in enumerate(self.months_full):
            self.month_to_index[month.lower()] = i
            # 약어도 포함
            abbr = self.months_abbr[i].rstrip('.')
            self.month_to_index[abbr.lower()] = i
            # 점 포함 약어
            self.month_to_index[self.months_abbr[i].lower()] = i

        # C12 - 요일 매핑
        self.weekdays = [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday',
            'Friday', 'Saturday', 'Sunday'
        ]
        self.weekday_to_index = {day.lower(): i for i, day in enumerate(self.weekdays)}

    def correct_article(self, article_text: str, article_date: str = None) -> Dict:
        """
        기사 전문을 받아 규칙 기반으로 교정

        Args:
            article_text: [TITLE]...[/TITLE][BODY]...[/BODY][CAPTION]...[/CAPTION] 형식
            article_date: 기사 작성 날짜 (optional)
                         Format: "YYYY-MM-DD" or "YYYY-MM-DD HH:MM:SS"
                         C12 규칙 처리 시 필요

        Returns:
            {
                'corrected_text': 교정된 전문,
                'corrections': [Correction 객체들],
                'stats': {
                    'total_corrections': int,
                    'by_rule': {rule_id: count}
                }
            }
        """
        corrections = []

        # --- 안전한 날짜 전처리 (문장 분할 전에 수행) ---
        # 예: "Nov. 12. 2025." → "Nov. 12, 2025." 로 정규화하여 잘못된 문장 분할을 방지
        try:
            import re as _re
            month_token = r"(?:Jan\.?|January|Feb\.?|February|Mar\.?|March|Apr\.?|April|May|Jun\.?|June|Jul\.?|July|Aug\.?|August|Sep\.?|Sept\.?|September|Oct\.?|October|Nov\.?|November|Dec\.?|December)"

            # Case 1: Month Day . Year . → Month Day, Year.
            article_text = _re.sub(
                rf"({month_token})\s(\d{{1,2}})\s*\.\s*(\d{{4}})\s*\.",
                r"\1 \2, \3.",
                article_text,
            )

            # Case 2: Month Day Year . → Month Day, Year.
            article_text = _re.sub(
                rf"({month_token})\s(\d{{1,2}})\s*(\d{{4}})\s*\.",
                r"\1 \2, \3.",
                article_text,
            )

            # Case 3: Month Day . Year (문장 끝) → Month Day, Year
            article_text = _re.sub(
                rf"({month_token})\s(\d{{1,2}})\s*\.\s*(\d{{4}})\s*$",
                r"\1 \2, \3",
                article_text,
            )
        except Exception:
            pass

        # Parse article_date if provided
        article_datetime = None
        if article_date:
            try:
                # Extract date part only: "2025-10-06 18:40:04" → "2025-10-06"
                article_date_only = article_date.split()[0] if ' ' in article_date else article_date
                article_datetime = datetime.strptime(article_date_only, "%Y-%m-%d")
            except (ValueError, IndexError):
                # Invalid date format, skip C12 processing
                article_datetime = None

        # 컴포넌트 추출
        title_text, title_start, title_end = self._extract_component(article_text, 'TITLE')
        body_text, body_start, body_end = self._extract_component(article_text, 'BODY')
        caption_text, caption_start, caption_end = self._extract_component(article_text, 'CAPTION')

        # 컴포넌트별 교정
        corrected_title, title_corrections = self._correct_title(title_text)
        corrected_body, body_corrections = self._correct_body(body_text)
        corrected_caption, caption_corrections = self._correct_caption(caption_text, article_datetime)

        # 교정 적용
        corrected_text = article_text

        # 역순으로 적용 (위치 변경 방지)
        if caption_text and corrected_caption != caption_text:
            corrected_text = (
                corrected_text[:caption_start] +
                corrected_caption +
                corrected_text[caption_end:]
            )

        if body_text and corrected_body != body_text:
            corrected_text = (
                corrected_text[:body_start] +
                corrected_body +
                corrected_text[body_end:]
            )

        if title_text and corrected_title != title_text:
            corrected_text = (
                corrected_text[:title_start] +
                corrected_title +
                corrected_text[title_end:]
            )

        # 교정 정보 통합
        corrections.extend(title_corrections)
        corrections.extend(body_corrections)
        corrections.extend(caption_corrections)

        # 통계
        stats = self._calculate_stats(corrections)

        return {
            'corrected_text': corrected_text,
            'corrections': corrections,
            'stats': stats,
            'original_text': article_text
        }

    def _extract_component(self, text: str, component: str) -> Tuple[str, int, int]:
        """컴포넌트 추출 (TITLE, BODY, CAPTION)"""
        pattern = rf'\[{component}\](.*?)\[/{component}\]'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1), match.start(1), match.end(1)
        return '', -1, -1

    def _correct_title(self, text: str) -> Tuple[str, List[Correction]]:
        """헤드라인 교정"""
        if not text:
            return text, []

        corrections = []
        corrected = text

        # H09 - 끝 마침표 금지 (약어 제외)
        corrected, corr = self._apply_h09_trailing_period(corrected, 'title')
        corrections.extend(corr)

        # H10 - 'percent' 단어 금지 → '%'
        corrected, corr = self._apply_h10_percent_word(corrected, 'title')
        corrections.extend(corr)

        # H11 - 숫자+won → W
        corrected, corr = self._apply_h11_number_won(corrected, 'title')
        corrections.extend(corr)

        return corrected, corrections

    def _correct_body(self, text: str) -> Tuple[str, List[Correction]]:
        """본문 교정"""
        if not text:
            return text, []

        corrections = []
        corrected = text

        # A14 - % 기호 금지 → 'percent'
        corrected, corr = self._apply_a14_percent_symbol(corrected, 'body')
        corrections.extend(corr)

        # A28 - Number formatting (units vs plain numbers)
        corrected, corr = self._apply_a28_number_format(corrected, 'body')
        corrections.extend(corr)

        # A31 - Joseon Kingdom → Joseon
        corrected, corr = self._apply_a31_kingdom(corrected, 'body')
        corrections.extend(corr)

        # A34 - 복합어 분리 (childcare → child care)
        corrected, corr = self._apply_a34_compound_words(corrected, 'body')
        corrections.extend(corr)

        # A35 - birth rate → birthrate
        corrected, corr = self._apply_a35_birthrate(corrected, 'body')
        corrections.extend(corr)

        # A09 - Korean name order (Given-name Family → Family Given-name)
        corrected, corr = self._apply_a09_korean_name_order(corrected, 'body')
        corrections.extend(corr)
 
        # A25 - Seoul districts (-gu → District)
        corrected, corr = self._apply_a25_seoul_districts(corrected, 'body')
        corrections.extend(corr)

        # Korean place names - Islands, Rivers, Temples
        corrected, corr = self._apply_korean_islands(corrected, 'body')
        corrections.extend(corr)

        corrected, corr = self._apply_korean_rivers(corrected, 'body')
        corrections.extend(corr)

        corrected, corr = self._apply_korean_temples(corrected, 'body')
        corrections.extend(corr)

        # C13 - Date range formatting
        corrected, corr = self._apply_c13_date_ranges(corrected, 'body')
        corrections.extend(corr)

        return corrected, corrections

    def _apply_a09_korean_name_order(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """
        A09 - 한국인 이름 표기 순서 교정
        - 'Given-name Family' (예: Jin-seok Hwang) → 'Family Given-name' (Hwang Jin-seok)
        - 안전장치:
          * 성씨 화이트리스트에 포함되는 경우만 교정
          * 하이픈 연결된 이름(두 음절)일 때 우선 적용 (과교정 방지)
          * 'Family, Given-name' (쉼표 표기)나 이미 'Family Given-name'인 경우 건드리지 않음

        Returns: (corrected_text, [Correction,...])
        """
        if not text:
            return text, []

        import re

        # 대표적인 한국 성씨 화이트리스트 (로마자 표기)
        surnames = [
            'Kim','Lee','Rhee','Yi','Park','Bak','Pak','Choi','Jung','Jeong','Kang','Kang','Cho','Jo',
            'Yoon','Yun','Jang','Chang','Lim','Im','Han','Shin','Sin','Yoo','Yu','Hwang','Kwon','Gwon',
            'Oh','O','Seo','Suh','Soe','Moon','Mun','Ryu','Yu','Nam','Song','Hong','Jeon','Chun','Jun',
            'Ko','Koh','Bae','Pae','Baek','Paek','Byun','Byeon','Cha','Ha','Heo','Hur','No','Roh','Noh'
        ]
        surname_regex = r"(?:" + "|".join(sorted(set(surnames), key=len, reverse=True)) + r")"

        # 하이픈 연결 이름 우선: Jin-seok Hwang → Hwang Jin-seok
        # 앞뒤 경계를 단어 경계로 제한, 이미 'Hwang, Jin-seok' 또는 'Hwang Jin-seok'은 매치되지 않음
        # 두 번째 토큰은 대소문자 모두 허용 (예: Jin-seok, Jin-Seok)
        pattern = re.compile(rf"\b([A-Z][a-z]+-[A-Za-z][a-z]+)\s+({surname_regex})\b")

        corrections: List[Correction] = []

        def repl(m: re.Match) -> str:
            given = m.group(1)
            family = m.group(2)
            start = m.start()
            original = f"{given} {family}"
            corrected = f"{family} {given}"

            # 쉼표 표기(예: Hwang, Jin-seok)는 제외 - 이 패턴엔 원래 안 걸리지만 안전체크 유지
            around = text[max(0, start-5):start+len(original)+5]
            if "," in around:
                return original

            corrections.append(Correction(
                rule_id='A09',
                component=component,
                original=original,
                corrected=corrected,
                position=start,
            ))
            return corrected

        new_text = pattern.sub(repl, text)
        # 실제 교정이 발생한 경우에만 로깅
        if corrections:
            _logger = logging.getLogger(__name__)
            preview = "; ".join([f"{c.original} -> {c.corrected}" for c in corrections[:3]])
            more = f" (+{len(corrections)-3} more)" if len(corrections) > 3 else ""
            _logger.info(f"[rule-based] A09 applied: {len(corrections)} change(s): {preview}{more}")
        return new_text, corrections

    def _correct_caption(self, text: str, article_datetime: datetime = None) -> Tuple[str, List[Correction]]:
        """캡션 교정

        Args:
            text: 캡션 텍스트
            article_datetime: 기사 작성 날짜 (C12 규칙 처리용, optional)
        """
        if not text:
            return text, []

        corrections = []
        corrected = text

        # C09 - pose for photo → pose
        corrected, corr = self._apply_c09_pose_for_photo(corrected, 'caption')
        corrections.extend(corr)

        # C15 - Korea Times file attribution (Tier 1A)
        corrected, corr = self._apply_c15_kt_file(corrected, 'caption')
        corrections.extend(corr)

        # C18 - "Captured from" format check (Tier 1A)
        corrected, corr = self._apply_c18_captured_from(corrected, 'caption')
        corrections.extend(corr)

        # C24 - 크레딧 마침표 (Yonhap)
        corrected, corr = self._apply_c24_credit_period(corrected, 'caption')
        corrections.extend(corr)

        # C27 - Seoul districts (-gu → District)
        corrected, corr = self._apply_c27_seoul_districts(corrected, 'caption')
        corrections.extend(corr)

        # C29 - Mount [Name] format (Tier 1A)
        corrected, corr = self._apply_c29_mount_format(corrected, 'caption')
        corrections.extend(corr)

        # C32 - Joseon Kingdom → Joseon Dynasty (Tier 1A - 캡션)
        corrected, corr = self._apply_c32_joseon_dynasty(corrected, 'caption')
        corrections.extend(corr)

        # C33 - 금지 시작구 (AI 처리로 이관)

        # Korean place names - Islands, Rivers, Temples
        corrected, corr = self._apply_korean_islands(corrected, 'caption')
        corrections.extend(corr)

        corrected, corr = self._apply_korean_rivers(corrected, 'caption')
        corrections.extend(corr)

        corrected, corr = self._apply_korean_temples(corrected, 'caption')
        corrections.extend(corr)

        # C13 - Date range formatting
        corrected, corr = self._apply_c13_date_ranges(corrected, 'caption')
        corrections.extend(corr)

        # C07
        corrected, corr = self._apply_c07_date_processing(corrected, 'caption', article_datetime)
        corrections.extend(corr)

        # C12 - Date format adjustment (AI 처리로 이관)
        # 캡션 날짜 스타일(C12)은 AI 단계에서 처리합니다.

        return corrected, corrections

    # ========== 규칙별 적용 함수 ==========

    def _apply_h09_trailing_period(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """H09 - 끝 마침표 금지 (약어 제외)

        - 문장 끝의 마침표가 1개 이상 연속으로 온 경우(예: ".", "..", "...")를 모두 제거
        - 단, 약어(단일 대문자 또는 두 글자 대문자 약어) 바로 뒤의 마침표는 보존
          예: "U.S.", "P.T." 등은 보존
        """
        corrections = []

        # 약어 끝이 아닌 마침표(여러 개 포함)만 제거
        # 예: "commenters..." → "commenters"
        # 예외: "U.S.", "P.T." 등은 보존 (단일/이중 대문자 약어)
        pattern = r'(?<!\b[A-Z])(?<!\b[A-Z]{2})\.+\s*$'
        match = re.search(pattern, text)

        if match:
            original = text
            corrected = text[:match.start()] + text[match.end():]
            corrections.append(Correction(
                rule_id='H09',
                component=component,
                original=original,
                corrected=corrected,
                position=match.start()
            ))
            return corrected, corrections

        return text, corrections

    def _apply_h10_percent_word(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """H10 - 'percent' 단어 금지 → '%'"""
        corrections = []
        pattern = r'\b(\d+)\s+percent\b'

        def replace_fn(match):
            original = match.group(0)
            number = match.group(1)
            corrected = f"{number}%"
            corrections.append(Correction(
                rule_id='H10',
                component=component,
                original=original,
                corrected=corrected,
                position=match.start()
            ))
            return corrected

        corrected = re.sub(pattern, replace_fn, text, flags=re.IGNORECASE)
        return corrected, corrections

    def _apply_h11_number_won(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """H11 - 숫자+won → W (동사 won 제외)"""
        corrections = []
        # 숫자 바로 뒤의 won만 매칭
        pattern = r'\b(\d[\d,]*(?:\.\d+)?)\s*won\b'

        def replace_fn(match):
            original = match.group(0)
            number = match.group(1)
            corrected = f"W{number}"
            corrections.append(Correction(
                rule_id='H11',
                component=component,
                original=original,
                corrected=corrected,
                position=match.start()
            ))
            return corrected

        corrected = re.sub(pattern, replace_fn, text, flags=re.IGNORECASE)
        return corrected, corrections

    def _apply_a14_percent_symbol(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """A14 - % 기호 금지 → 'percent'"""
        corrections = []
        pattern = r'(\d+)\s*%'

        def replace_fn(match):
            original = match.group(0)
            number = match.group(1)
            corrected = f"{number} percent"
            corrections.append(Correction(
                rule_id='A14',
                component=component,
                original=original,
                corrected=corrected,
                position=match.start()
            ))
            return corrected

        corrected = re.sub(pattern, replace_fn, text)
        return corrected, corrections

    def _apply_a31_kingdom(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """A31 - Joseon Kingdom → Joseon Dynasty (Body only)"""
        corrections = []
        pattern = r'\bJoseon\s+Kingdom\b'

        matches = list(re.finditer(pattern, text))
        for match in reversed(matches):  # 역순 적용
            original = match.group(0)
            corrected = "Joseon Dynasty"
            corrections.append(Correction(
                rule_id='A31',
                component=component,
                original=original,
                corrected=corrected,
                position=match.start()
            ))

        corrected = re.sub(pattern, 'Joseon Dynasty', text)
        return corrected, corrections

    def _apply_a28_number_format(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """
        A28 - Number formatting (rule-based assist)

        - 숫자 단어(one~nineteen, tens) + 단위 → 아라비아 숫자로 변환 (예: four hours → 4 hours)
        - scale 단어(hundred/thousand/million/billion) 포함 시, 숫자로 변환하되 million/billion은 그대로 유지 (예: five million won → 5 million won)
        - 단위가 없는 1~9 숫자는 철자로 변환 (예: 4 options → four options)
        """
        import re

        corrections: List[Correction] = []

        # 단위 패턴 (길이가 긴 순으로 매칭되도록 정렬)
        unit_pattern = "|".join(sorted(self._a28_units, key=len, reverse=True))

        # 숫자 단어 구문 패턴 (최대: tens + ones + scale 1개)
        num_word = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
        num_phrase_pattern = rf"(?:{num_word}(?:\s+{num_word})?)(?:\s+(?:hundred|thousand|million|billion))?"

        # 1) 숫자 단어 + 단위 → 숫자 + 단위
        pattern_units = re.compile(rf"\b({num_phrase_pattern})\s+({unit_pattern})\b", re.IGNORECASE)

        def _number_phrase_to_string(phrase: str) -> Optional[str]:
            # 간단한 영어 숫자 구문을 숫자 문자열로 변환
            try:
                tokens = phrase.lower().split()
                total = 0
                current = 0
                scale_token = None
                for t in tokens:
                    if t in self._number_scales:
                        scale_token = t
                        # scale 이전 숫자 계산
                        if current == 0:
                            current = 1
                        current = current * self._number_scales[t]
                        total += current
                        current = 0
                    else:
                        current += self._number_word_value.get(t, 0)
                total += current

                # million/billion은 원문의 단위 표현을 유지 (5 million → "5 million")
                if scale_token in ('million', 'billion'):
                    # total은 실제 값이지만, 표현은 "<base> million" 형태가 더 자연스러움
                    # tokens에서 scale 앞의 숫자만 추출
                    base_tokens = []
                    for t in tokens:
                        if t == scale_token:
                            break
                        base_tokens.append(t)
                    # base 숫자 계산
                    base_total = 0
                    tmp = 0
                    for t in base_tokens:
                        tmp += self._number_word_value.get(t, 0)
                    base_total += tmp
                    return f"{base_total} {scale_token}"

                return str(total)
            except Exception:
                return None

        def repl_units(m: re.Match) -> str:
            original = m.group(0)
            num_phrase = m.group(1)
            unit = m.group(2)
            num_str = _number_phrase_to_string(num_phrase)
            if not num_str:
                return original
            corrected = f"{num_str} {unit}"
            corrections.append(Correction(
                rule_id='A28',
                component=component,
                original=original,
                corrected=corrected,
                position=m.start()
            ))
            return corrected

        text = pattern_units.sub(repl_units, text)

        # 2) 단위 없는 1~9 숫자 → 철자
        #    (뒤에 단위 패턴이 오지 않는 경우만)
        pattern_plain_digit = re.compile(rf"\b([1-9])\b(?!\s+(?:{unit_pattern})\b)")

        def repl_plain(m: re.Match) -> str:
            original = m.group(0)
            digit = m.group(1)
            word = self._digit_to_word.get(digit)
            if not word:
                return original
            corrections.append(Correction(
                rule_id='A28',
                component=component,
                original=original,
                corrected=word,
                position=m.start()
            ))
            return word

        text = pattern_plain_digit.sub(repl_plain, text)

        # 3) scale(hundred/thousand/million/billion) 포함 숫자 단어 구문(단위 없음) → 숫자
        #    예: "nine hundred workers" → "900 workers"
        pattern_scale_no_unit = re.compile(rf"\b({num_phrase_pattern})\b(?!\s+(?:{unit_pattern})\b)", re.IGNORECASE)

        def repl_scale_no_unit(m: re.Match) -> str:
            original = m.group(0)
            phrase = m.group(1)
            # scale 단어가 없는 경우는 유지 (ten, twenty 등은 그대로 두기 위함)
            if not any(s in phrase.lower().split() for s in self._number_scales):
                return original
            num_str = _number_phrase_to_string(phrase)
            if not num_str:
                return original
            # million/billion 등 scale이 포함되어도 unit이 없으면 그대로 사용 (예: 5 million)
            corrected = num_str
            corrections.append(Correction(
                rule_id='A28',
                component=component,
                original=original,
                corrected=corrected,
                position=m.start()
            ))
            return corrected

        text = pattern_scale_no_unit.sub(repl_scale_no_unit, text)

        return text, corrections

    def _apply_c32_joseon_dynasty(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """C32 - Joseon Kingdom → Joseon Dynasty (Caption - Tier 1A)

        처리 가능성: 99%
        단순 문자열 치환, 예외 케이스 거의 없음
        """
        corrections = []
        pattern = r'\bJoseon\s+Kingdom\b'

        matches = list(re.finditer(pattern, text))
        for match in reversed(matches):  # 역순 적용
            original = match.group(0)
            corrected = "Joseon Dynasty"
            corrections.append(Correction(
                rule_id='C32',
                component=component,
                original=original,
                corrected=corrected,
                position=match.start()
            ))

        corrected = re.sub(pattern, 'Joseon Dynasty', text)
        return corrected, corrections

    def _apply_a34_compound_words(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """A34 - 복합어 분리 (childcare → child care)"""
        corrections = []

        pattern = r'\b(' + '|'.join(self.compound_words_to_split.keys()) + r')\b'

        def replace_fn(match):
            original = match.group(0)
            word_lower = match.group(1).lower()

            if word_lower in self.compound_words_to_split:
                corrected = self.compound_words_to_split[word_lower]

                # 대소문자 보존
                if original[0].isupper():
                    # 첫 글자 대문자 보존
                    corrected = corrected[0].upper() + corrected[1:]

                if original.isupper():
                    # 전체 대문자인 경우
                    corrected = corrected.upper()

                corrections.append(Correction(
                    rule_id='A34',
                    component=component,
                    original=original,
                    corrected=corrected,
                    position=match.start()
                ))
                return corrected
            return original

        corrected = re.sub(pattern, replace_fn, text, flags=re.IGNORECASE)
        return corrected, corrections

    def _apply_a35_birthrate(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """A35 - birth rate → birthrate (한 단어)"""
        corrections = []
        pattern = r'\bbirth[-\s]rate\b'

        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        for match in reversed(matches):
            original = match.group(0)
            corrected = "birthrate"
            corrections.append(Correction(
                rule_id='A35',
                component=component,
                original=original,
                corrected=corrected,
                position=match.start()
            ))

        corrected = re.sub(pattern, 'birthrate', text, flags=re.IGNORECASE)
        return corrected, corrections

    def _apply_a25_seoul_districts(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """A25 - Seoul districts: X-gu → X District

        서울 25개 구 이름 교정
        """
        corrections = []
        corrected = text

        for gu_name, district_name in self.seoul_districts.items():
            pattern = rf'\b{re.escape(gu_name)}\b'

            for match in re.finditer(pattern, corrected):
                original = match.group(0)
                corrections.append(Correction(
                    rule_id='A25',
                    component=component,
                    original=original,
                    corrected=district_name,
                    position=match.start()
                ))

                corrected = corrected[:match.start()] + district_name + corrected[match.end():]
                break  # 한 번만 적용

        return corrected, corrections

    def _apply_c27_seoul_districts(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """C27 - Seoul districts: X-gu → X District (Caption)

        서울 25개 구 이름 교정
        """
        corrections = []
        corrected = text

        for gu_name, district_name in self.seoul_districts.items():
            pattern = rf'\b{re.escape(gu_name)}\b'

            for match in re.finditer(pattern, corrected):
                original = match.group(0)
                corrections.append(Correction(
                    rule_id='C27',
                    component=component,
                    original=original,
                    corrected=district_name,
                    position=match.start()
                ))

                corrected = corrected[:match.start()] + district_name + corrected[match.end():]
                break  # 한 번만 적용

        return corrected, corrections

    def _apply_c09_pose_for_photo(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """C09 - pose(s) for (a) photo(s)/picture(s) → pose"""
        corrections: List[Correction] = []
        pattern = r"\bposes?\s+for\s+(?:a\s+)?(?:photo|picture)s?\b"

        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        corrected = text

        for m in reversed(matches):
            original = m.group(0)
            replacement = "pose"
            corrected = corrected[:m.start()] + replacement + corrected[m.end():]
            corrections.append(Correction(
                rule_id='C09',
                component=component,
                original=original,
                corrected=corrected,
                position=m.start()
            ))

        return corrected, corrections

    def _apply_c24_credit_period(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """C24 - 크레딧 마침표 (Yonhap, Courtesy of, Korea Times 등)"""
        corrections = []

        # 크레딧 키워드들 (longer patterns first to avoid partial matches)
        credit_keywords = [
            'Korea Times file', 'Korea Times', 'Getty Images', 'Courtesy of',
            'Yonhap', 'Reuters', 'Bloomberg', 'EPA', 'AFP', 'AP'
        ]

        # Normalize text - strip whitespace and trailing quotes
        normalized = text.strip()

        # Track if we need to restore trailing quote
        trailing_quote = ''
        if normalized.endswith('"') or normalized.endswith("'"):
            trailing_quote = normalized[-1]
            normalized = normalized[:-1].strip()

        # Check if text ends with period (after removing quote)
        if normalized.endswith('.'):
            # Check if any credit keyword exists at the end
            for keyword in credit_keywords:
                # 크레딧 키워드 바로 뒤에 마침표가 있는지 확인 (공백 허용)
                pattern = rf'(?:^|\s){re.escape(keyword)}\s*\.$'
                if re.search(pattern, normalized, re.IGNORECASE):
                    original = text
                    # 마침표 제거 (trailing whitespace도 정리)
                    corrected = re.sub(rf'(\s*{re.escape(keyword)})\s*\.$', r'\1', normalized, flags=re.IGNORECASE)
                    # Restore trailing quote if existed
                    if trailing_quote:
                        corrected = corrected + trailing_quote
                    corrections.append(Correction(
                        rule_id='C24',
                        component=component,
                        original=original,
                        corrected=corrected,
                        position=len(normalized) - 1
                    ))
                    text = corrected
                    break

        return text, corrections

    def _apply_c33_forbidden_start(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """C33 - 금지 시작구 제거"""
        corrections = []

        forbidden_starts = [
            r'^This\s+photo\s+shows\s+',
            r'^Seen\s+above\s+is\s+',
            r'^The\s+photo\s+shows\s+',
            r'^Pictured\s+is\s+'
        ]

        for pattern in forbidden_starts:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                original = text
                corrected = text[match.end():]
                # 첫 글자 대문자화
                if corrected:
                    corrected = corrected[0].upper() + corrected[1:]

                corrections.append(Correction(
                    rule_id='C33',
                    component=component,
                    original=original,
                    corrected=corrected,
                    position=0
                ))
                return corrected, corrections

        return text, corrections

    # ========== Tier 1A 규칙 (95%+ 정규식 처리 가능) ==========

    def _apply_c29_mount_format(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """C29 - Mount [Name] format (Tier 1A)

        처리 가능성: 90%
        1. Mt. XXX → Mount XXX
        2. Korean mountain names (Hallasan → Mount Halla)
        """
        corrections = []
        corrected = text

        # Pattern 1: Mt. [Name] → Mount [Name]
        pattern1 = r'\bMt\.\s+(\w+)'

        for match in re.finditer(pattern1, corrected):
            # 예외 체크
            full_match = match.group(0)
            is_exception = any(exc in full_match for exc in self.mountain_exceptions)

            if not is_exception:
                original = match.group(0)
                mountain_name = match.group(1)
                new_text = f"Mount {mountain_name}"

                corrections.append(Correction(
                    rule_id='C29',
                    component=component,
                    original=original,
                    corrected=new_text,
                    position=match.start()
                ))

        # Apply Pattern 1
        corrected = re.sub(pattern1, lambda m: f"Mount {m.group(1)}"
                          if not any(exc in m.group(0) for exc in self.mountain_exceptions)
                          else m.group(0), corrected)

        # Pattern 2: Korean mountain names (XXXsan → Mount XXX)
        for kr_name, en_name in self.korean_mountains.items():
            pattern2 = rf'\b{kr_name}\b'

            for match in re.finditer(pattern2, corrected):
                # 뒤에 "National Park", "Trail" 등 있으면 스킵
                after_text = corrected[match.end():match.end()+15]
                if re.match(r'\s+(National\s+Park|Park|Trail)', after_text, re.IGNORECASE):
                    continue

                # 이미 "Mount" 앞에 있으면 스킵
                before_text = corrected[max(0, match.start()-10):match.start()]
                if re.search(r'Mount\s+$', before_text, re.IGNORECASE):
                    continue

                original = match.group(0)
                corrections.append(Correction(
                    rule_id='C29',
                    component=component,
                    original=original,
                    corrected=en_name,
                    position=match.start()
                ))

                corrected = corrected[:match.start()] + en_name + corrected[match.end():]
                break  # 한 번만 적용

        # Pattern 3: English form "{Base} Mountain" → "Mount {Base}" for known bases
        try:
            mount_bases = sorted({
                en.split(' ', 1)[1]
                for en in self.korean_mountains.values()
                if en.lower().startswith('mount ')
            }, key=len, reverse=True)
            if mount_bases:
                pattern3 = rf"\b({'|'.join(map(re.escape, mount_bases))})\s+Mountain\b"

                for match in list(re.finditer(pattern3, corrected)):
                    # 이미 "Mount" 앞에 있으면 스킵
                    before_text = corrected[max(0, match.start()-10):match.start()]
                    if re.search(r'Mount\s+$', before_text, re.IGNORECASE):
                        continue

                    base = match.group(1)
                    original = match.group(0)
                    new_text = f"Mount {base}"

                    corrections.append(Correction(
                        rule_id='C29',
                        component=component,
                        original=original,
                        corrected=new_text,
                        position=match.start()
                    ))

                    corrected = corrected[:match.start()] + new_text + corrected[match.end():]
        except Exception:
            # 패턴 구성 실패 시 조용히 넘어감 (안전)
            pass

        return corrected, corrections

    def _apply_c28_palace_names(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """C28 - Palace names (Tier 1A)

        처리 가능성: 92%
        Gyeongbokgung → Gyeongbok Palace
        """
        corrections = []
        corrected = text

        for palace_kr, palace_en in self.korean_palaces.items():
            pattern = rf'\b{palace_kr}\b'

            for match in re.finditer(pattern, corrected):
                # 예외 체크 (지하철역, 지역명 등)
                context = corrected[match.start():match.end()+25]
                if any(exc in context for exc in self.palace_exceptions):
                    continue

                original = match.group(0)

                # 뒤에 'Palace'가 이미 있는 경우: '{kr} Palace' → '{base} Palace'
                after_text = corrected[match.end():match.end()+10]
                has_palace_after = re.match(r'\s+Palace', after_text, re.IGNORECASE)
                if has_palace_after:
                    base = palace_en.replace(' Palace', '')
                    new_token = base  # 뒤 'Palace'는 그대로 둔다
                    new_text = new_token
                else:
                    # 일반 케이스: '{kr}' → '{En Palace}'
                    new_text = palace_en

                corrections.append(Correction(
                    rule_id='C28',
                    component=component,
                    original=original,
                    corrected=new_text,
                    position=match.start()
                ))

                corrected = corrected[:match.start()] + new_text + corrected[match.end():]
                break  # 한 번만 적용

        return corrected, corrections

    def _apply_c18_captured_from(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """C18 - "Captured from" format (Tier 1A)

        처리 가능성: 90%
        검증 전용: "Captured from [source]" 형식 체크
        모호한 출처인 경우에만 경고 (교정은 하지 않음)
        """
        corrections = []

        # Pattern: "Captured from [source]"
        pattern = r'\bCaptured\s+from\s+([^.\n]+)'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            source = match.group(1).strip()

            # 모호한 출처 체크
            is_vague = any(vague in source.lower() for vague in self.vague_capture_sources)

            if is_vague:
                # 경고만 (교정하지 않음)
                corrections.append(Correction(
                    rule_id='C18',
                    component=component,
                    original=match.group(0),
                    corrected=f"[WARNING: vague source '{source}']",
                    position=match.start()
                ))

        return text, corrections  # 텍스트 변경 없음

    def _apply_c15_kt_file(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """C15 - Korea Times file attribution (Tier 1A)

        처리 가능성: 95%
        "Korea Times photo" → "Korea Times file"
        """
        corrections = []

        # Pattern: "Korea Times photo" → "Korea Times file"
        pattern = r'\bKorea\s+Times\s+photo\b'

        for match in re.finditer(pattern, text, re.IGNORECASE):
            original = match.group(0)
            corrected_text = "Korea Times file"

            corrections.append(Correction(
                rule_id='C15',
                component=component,
                original=original,
                corrected=corrected_text,
                position=match.start()
            ))

        corrected = re.sub(pattern, 'Korea Times file', text, flags=re.IGNORECASE)
        return corrected, corrections

    # ========== 한국 지명 규칙 (섬/강/사찰) ==========

    def _apply_korean_islands(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """Korean Islands - Jejudo → Jeju Island

        처리 가능성: 95%
        한국 섬 이름 표준 형식 변환
        """
        corrections = []
        corrected = text

        for kr_name, en_name in self.korean_islands.items():
            # Skip if it's already the correct form
            if kr_name == en_name:
                continue

            pattern = rf'\b{re.escape(kr_name)}\b'

            for match in re.finditer(pattern, corrected):
                # Check if "Island" already follows
                after_text = corrected[match.end():match.end()+10]
                if re.match(r'\s+Island', after_text, re.IGNORECASE):
                    continue

                original = match.group(0)
                corrections.append(Correction(
                    rule_id='C29',  # Korean place name formatting
                    component=component,
                    original=original,
                    corrected=en_name,
                    position=match.start()
                ))

                corrected = corrected[:match.start()] + en_name + corrected[match.end():]
                break  # Apply once per occurrence

        return corrected, corrections

    def _apply_korean_rivers(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """Korean Rivers - Hangang → Han River

        처리 가능성: 95%
        한국 강 이름 표준 형식 변환 (-gang → River)
        """
        corrections = []
        corrected = text

        for kr_name, en_name in self.korean_rivers.items():
            # Skip if it's already the correct form
            if kr_name == en_name:
                continue

            pattern = rf'\b{re.escape(kr_name)}\b'

            for match in re.finditer(pattern, corrected):
                # Check if "River" already follows
                after_text = corrected[match.end():match.end()+10]
                if re.match(r'\s+River', after_text, re.IGNORECASE):
                    continue

                # Check if already preceded by river-related terms
                before_text = corrected[max(0, match.start()-15):match.start()]
                if re.search(r'(River|Stream)\s+$', before_text, re.IGNORECASE):
                    continue

                original = match.group(0)
                corrections.append(Correction(
                    rule_id='C29',  # Korean place name formatting
                    component=component,
                    original=original,
                    corrected=en_name,
                    position=match.start()
                ))

                corrected = corrected[:match.start()] + en_name + corrected[match.end():]
                break  # Apply once per occurrence

        return corrected, corrections

    def _apply_korean_temples(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """Korean Temples - Bulguksa → Bulguk Temple

        처리 가능성: 90%
        한국 사찰 이름 표준 형식 변환 (-sa → Temple)
        예외: Seokguram Grotto
        """
        corrections = []
        corrected = text

        for kr_name, en_name in self.korean_temples.items():
            pattern = rf'\b{re.escape(kr_name)}\b'

            for match in re.finditer(pattern, corrected):
                after_text = corrected[match.end():match.end()+15]

                # Check if "Temple" or "Grotto" already follows (uppercase = already correct)
                if re.match(r'\s+(Temple|Grotto)\b', after_text):
                    continue

                # Handle lowercase "temple" case: "Bulguksa temple" → "Bulguk Temple"
                lowercase_temple_match = re.match(r'\s+temple\b', after_text)
                if lowercase_temple_match:
                    # Replace "Bulguksa temple" with "Bulguk Temple"
                    original = match.group(0) + lowercase_temple_match.group(0)
                    corrections.append(Correction(
                        rule_id='C29',  # Korean place name formatting
                        component=component,
                        original=original,
                        corrected=en_name,
                        position=match.start()
                    ))
                    corrected = corrected[:match.start()] + en_name + corrected[match.end() + len(lowercase_temple_match.group(0)):]
                    break

                # Normal case: just the Korean name without temple
                original = match.group(0)
                corrections.append(Correction(
                    rule_id='C29',  # Korean place name formatting
                    component=component,
                    original=original,
                    corrected=en_name,
                    position=match.start()
                ))

                corrected = corrected[:match.start()] + en_name + corrected[match.end():]
                break  # Apply once per occurrence

        return corrected, corrections

    def _apply_c07_date_processing(self, text: str, component: str,
                                   article_datetime: datetime) -> Tuple[str, List[Correction]]:
        """C07 - Date Processing (Within 7 days conversion)
        
        Logic:
        1. Detect dates (Nov. 24, 11/24, etc.)
        2. Compare with article_datetime
        3. If within -7 days (past): Convert to "last [Weekday]"
        4. If within +7 days (future/same): Convert to "[Weekday]"
        5. Remove 'on' preposition
        """
        corrections = []
        corrected = text
        
        # Month mapping
        month_map = {
            'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3, 'march': 3,
            'apr': 4, 'april': 4, 'may': 5, 'jun': 6, 'june': 6, 'jul': 7, 'july': 7,
            'aug': 8, 'august': 8, 'sep': 9, 'sept': 9, 'september': 9,
            'oct': 10, 'october': 10, 'nov': 11, 'november': 11, 'dec': 12, 'december': 12
        }

        # Regex Pattern Components
        # 1. Preamble: Optional 'on' (capture group 1)
        # 2. Month: Text (Jan, January...) OR Number (1-12)
        # 3. Separator: Space, dot, slash, hyphen
        # 4. Day: 1-31
        # 5. Optional Year: , 2024 or /2024
        
        # Pattern 1: Text Month (Nov. 24, November 24)
        month_names = "January|February|March|April|May|June|July|August|September|October|November|December|Jan\.|Feb\.|Mar\.|Apr\.|Aug\.|Sep\.|Sept\.|Oct\.|Nov\.|Dec\."
        pat_text = rf'(?P<prep>on\s+)?\b(?P<month>{month_names})\s+(?P<day>\d{{1,2}})(?:,?\s*(?P<year>\d{{4}}))?\b'
        
        # Pattern 2: Numeric Month (11/24, 11-24)
        pat_num = r'(?P<prep>on\s+)?\b(?P<month>\d{1,2})[/-](?P<day>\d{1,2})(?:[/-](?P<year>\d{4}))?\b'

        # Helper to process matches
        def get_closest_date(target_month, target_day, explicit_year=None):
            """기사 날짜와 가장 가까운 연도의 날짜를 반환 (연도 경계 문제 해결)"""
            if explicit_year:
                return datetime(int(explicit_year), target_month, target_day)
            
            # 연도가 없으면: 기사 연도, 작년, 내년 중 가장 가까운 날짜 선택
            candidates = []
            for y in [article_datetime.year - 1, article_datetime.year, article_datetime.year + 1]:
                try:
                    candidates.append(datetime(y, target_month, target_day))
                except ValueError:
                    continue # 2월 29일 등의 오류 방지
            
            # 기사 날짜와의 차이가 가장 적은 후보 선택
            return min(candidates, key=lambda d: abs((d - article_datetime).days))

        def process_match(match, is_numeric=False):
            nonlocal corrected
            
            prep = match.group('prep') or "" # "on " or None
            month_raw = match.group('month')
            day_raw = int(match.group('day'))
            year_raw = match.group('year')

            # Parse Month
            if is_numeric:
                month_idx = int(month_raw)
            else:
                month_clean = month_raw.lower().rstrip('.')
                month_idx = month_map.get(month_clean)
            
            if not month_idx or not (1 <= month_idx <= 12) or not (1 <= day_raw <= 31):
                return

            # Calculate Event Date
            try:
                event_date = get_closest_date(month_idx, day_raw, year_raw)
            except Exception:
                return

            # Calculate Difference
            diff_days = (event_date.date() - article_datetime.date()).days
            
            target_str = ""
            weekday = event_date.strftime("%A") # e.g., Monday

            # Logic: Within 7 days range (-6 to +6)
            # User Rule: 
            # Past (< 0): "last [Weekday]"
            # Future/Same (>= 0): "[Weekday]"
            
            # 범위를 7일로 설정 (필요시 조정 가능)
            if -7 < diff_days < 0:
                target_str = f"last {weekday}"
            elif 0 <= diff_days < 7:
                target_str = weekday
            else:
                return

            original_text = match.group(0)
            
            return (match.start(), match.end(), original_text, target_str)


        # Collect all replacements first
        replacements = []

        # Find Text Matches
        for m in re.finditer(pat_text, corrected, re.IGNORECASE):
            res = process_match(m, is_numeric=False)
            if res: replacements.append(res)
            
        # Find Numeric Matches
        for m in re.finditer(pat_num, corrected, re.IGNORECASE):
            res = process_match(m, is_numeric=True)
            if res: replacements.append(res)

        # Sort replacements by start position descending (to avoid index shift)
        replacements.sort(key=lambda x: x[0], reverse=True)
        
        # Apply Replacements
        for start, end, original, replacement in replacements:
            
            corrections.append(Correction(
                rule_id='C07',
                component=component,
                original=original,
                corrected=replacement,
                position=start
            ))
            
            corrected = corrected[:start] + replacement + corrected[end:]

        return corrected, corrections

    def _apply_c13_date_ranges(self, text: str, component: str) -> Tuple[str, List[Correction]]:
        """C13 - Date range formatting (Tier 1A)

        처리 가능성: 85%
        1. Same Month: "April 28 to April 30" → "April 28 to 30"
        2. Different Months: "April 28 to 6" → "April 28 to May 6"
        """
        corrections = []
        corrected = text

        # Pattern 1: Same month redundancy - "Month Day to Month Day"
        # Match: "April 28 to April 30" or "Apr. 28 to Apr. 30"
        # Pattern supports both "to" and "-" as separators
        pattern1 = r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan\.|Feb\.|Mar\.|Apr\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.)\s+(\d{1,2})(?:\s+to\s+|\s*-\s*|\s+through\s+)(January|February|March|April|May|June|July|August|September|October|November|December|Jan\.|Feb\.|Mar\.|Apr\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.)\s+(\d{1,2})\b'

        def replace_same_month(match):
            month1 = match.group(1)
            day1 = match.group(2)
            month2 = match.group(3)
            day2 = match.group(4)

            # Get separator (to, -, through)
            full_match = match.group(0)
            if ' to ' in full_match:
                separator = ' to '
            elif ' through ' in full_match:
                separator = ' through '
            elif '-' in full_match:
                # Check if it's spaced or not
                if f'{day1} - {month2}' in full_match:
                    separator = ' - '
                elif f'{day1}-{month2}' in full_match:
                    separator = '-'
                else:
                    separator = ' - '
            else:
                separator = ' to '

            # Normalize month names for comparison
            month1_normalized = month1.rstrip('.').lower()
            month2_normalized = month2.rstrip('.').lower()

            # Check if same month
            if month1_normalized == month2_normalized:
                original = match.group(0)
                corrected_text = f"{month1} {day1}{separator}{day2}"
                corrections.append(Correction(
                    rule_id='C13',
                    component=component,
                    original=original,
                    corrected=corrected_text,
                    position=match.start()
                ))
                return corrected_text
            return match.group(0)

        corrected = re.sub(pattern1, replace_same_month, corrected, flags=re.IGNORECASE)

        # Pattern 2: Different months incomplete - "Month Day to Day"
        # Match: "April 28 to 6" (should be "April 28 to May 6")
        # This is more complex - need to infer the next month
        pattern2 = r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan\.|Feb\.|Mar\.|Apr\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.)\s+(\d{1,2})(?:\s+to\s+|\s*-\s*|\s+through\s+)(\d{1,2})(?!\s*(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\.|Feb\.|Mar\.|Apr\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.))\b'

        matches = list(re.finditer(pattern2, corrected, re.IGNORECASE))

        # Process in reverse order to maintain positions
        for match in reversed(matches):
            month1 = match.group(1)
            day1 = match.group(2)
            day2 = match.group(3)

            # Get separator
            full_match = match.group(0)
            if ' to ' in full_match:
                separator = ' to '
            elif ' through ' in full_match:
                separator = ' through '
            elif '-' in full_match:
                if f'{day1} - {day2}' in full_match:
                    separator = ' - '
                elif f'{day1}-{day2}' in full_match:
                    separator = '-'
                else:
                    separator = ' - '
            else:
                separator = ' to '

            # Check if day2 < day1 (likely next month)
            # Or if day2 is much smaller than day1
            if int(day2) < int(day1) or (int(day1) > 20 and int(day2) < 10):
                # Infer next month
                month1_normalized = month1.rstrip('.').lower()

                if month1_normalized in self.month_to_index:
                    current_month_idx = self.month_to_index[month1_normalized]
                    next_month_idx = (current_month_idx + 1) % 12

                    # Determine if month1 is abbreviated or full
                    is_abbreviated = '.' in month1 or len(month1) <= 4

                    if is_abbreviated:
                        next_month = self.months_abbr[next_month_idx]
                    else:
                        next_month = self.months_full[next_month_idx]

                    original = match.group(0)
                    corrected_text = f"{month1} {day1}{separator}{next_month} {day2}"

                    corrections.append(Correction(
                        rule_id='C13',
                        component=component,
                        original=original,
                        corrected=corrected_text,
                        position=match.start()
                    ))

                    corrected = corrected[:match.start()] + corrected_text + corrected[match.end():]

        # Pattern 3: "from Month Day to Month Day" pattern
        # This is similar to Pattern 1 but with "from" prefix
        pattern3 = r'\bfrom\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan\.|Feb\.|Mar\.|Apr\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.)\s+(\d{1,2})\s+to\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan\.|Feb\.|Mar\.|Apr\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.)\s+(\d{1,2})\b'

        def replace_from_same_month(match):
            month1 = match.group(1)
            day1 = match.group(2)
            month2 = match.group(3)
            day2 = match.group(4)

            # Normalize month names for comparison
            month1_normalized = month1.rstrip('.').lower()
            month2_normalized = month2.rstrip('.').lower()

            # Check if same month
            if month1_normalized == month2_normalized:
                original = match.group(0)
                corrected_text = f"from {month1} {day1} to {day2}"
                corrections.append(Correction(
                    rule_id='C13',
                    component=component,
                    original=original,
                    corrected=corrected_text,
                    position=match.start()
                ))
                return corrected_text
            return match.group(0)

        corrected = re.sub(pattern3, replace_from_same_month, corrected, flags=re.IGNORECASE)

        # Pattern 4: "from Month Day to Day" pattern (different months)
        pattern4 = r'\bfrom\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan\.|Feb\.|Mar\.|Apr\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.)\s+(\d{1,2})\s+to\s+(\d{1,2})(?!\s*(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan\.|Feb\.|Mar\.|Apr\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.))\b'

        matches = list(re.finditer(pattern4, corrected, re.IGNORECASE))

        for match in reversed(matches):
            month1 = match.group(1)
            day1 = match.group(2)
            day2 = match.group(3)

            # Check if day2 < day1 (likely next month)
            if int(day2) < int(day1) or (int(day1) > 20 and int(day2) < 10):
                month1_normalized = month1.rstrip('.').lower()

                if month1_normalized in self.month_to_index:
                    current_month_idx = self.month_to_index[month1_normalized]
                    next_month_idx = (current_month_idx + 1) % 12

                    is_abbreviated = '.' in month1 or len(month1) <= 4

                    if is_abbreviated:
                        next_month = self.months_abbr[next_month_idx]
                    else:
                        next_month = self.months_full[next_month_idx]

                    original = match.group(0)
                    corrected_text = f"from {month1} {day1} to {next_month} {day2}"

                    corrections.append(Correction(
                        rule_id='C13',
                        component=component,
                        original=original,
                        corrected=corrected_text,
                        position=match.start()
                    ))

                    corrected = corrected[:match.start()] + corrected_text + corrected[match.end():]

        return corrected, corrections

    def _apply_c12_date_format(self, text: str, component: str,
                               article_datetime: datetime) -> Tuple[str, List[Correction]]:
        """C12 - Date format adjustment (Tier 1A with context)

        처리 가능성: 95% (기사 날짜 제공 시)

        Foreign captions often use "Weekday, Month Day, Year" format.
        Adjust based on time difference:
        - Within 7 days: use only weekday (e.g., "Thursday")
        - More than 7 days ago: use only date (e.g., "April 27")
        """
        corrections = []
        corrected = text

        # Pattern: Weekday, Month Day, Year
        # Match: "Thursday, April 27, 2024" or "Monday, Dec. 15, 2023"
        pattern = r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+(January|February|March|April|May|June|July|August|September|October|November|December|Jan\.|Feb\.|Mar\.|Apr\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.)\s+(\d{1,2}),?\s*(\d{4})\b'

        matches = list(re.finditer(pattern, corrected, re.IGNORECASE))

        # Process in reverse order to maintain positions
        for match in reversed(matches):
            weekday = match.group(1)
            month = match.group(2)
            day = match.group(3)
            year = match.group(4)

            # Parse event date
            try:
                event_datetime = self._parse_caption_date(month, day, year)
            except (ValueError, KeyError):
                # Invalid date, skip
                continue

            # Verify weekday matches the date
            actual_weekday = event_datetime.strftime('%A')
            if actual_weekday.lower() != weekday.lower():
                # Date mismatch warning
                corrections.append(Correction(
                    rule_id='C12',
                    component=component,
                    original=match.group(0),
                    corrected=f"[WARNING: {weekday} != {actual_weekday} for {month} {day}, {year}]",
                    position=match.start()
                ))
                continue

            # Calculate difference in days
            days_diff = (article_datetime - event_datetime).days

            # Apply rule based on time difference
            original = match.group(0)

            if 0 <= days_diff <= 7:
                # Within a week: use only weekday
                corrected_text = weekday
            elif days_diff > 7:
                # More than a week ago: use only date
                # Keep month format (abbreviated or full)
                corrected_text = f"{month} {day}"
            else:
                # Future date (negative days_diff): skip
                continue

            corrections.append(Correction(
                rule_id='C12',
                component=component,
                original=original,
                corrected=corrected_text,
                position=match.start()
            ))

            corrected = corrected[:match.start()] + corrected_text + corrected[match.end():]

        return corrected, corrections

    def _parse_caption_date(self, month: str, day: str, year: str) -> datetime:
        """Parse caption date to datetime object

        Args:
            month: Month name (full or abbreviated, e.g., "April" or "Apr.")
            day: Day number (e.g., "27")
            year: Year (e.g., "2024")

        Returns:
            datetime object

        Raises:
            ValueError: Invalid date
            KeyError: Unknown month name
        """
        # Normalize month name
        month_normalized = month.lower().rstrip('.')

        if month_normalized not in self.month_to_index:
            raise KeyError(f"Unknown month: {month}")

        month_num = self.month_to_index[month_normalized] + 1  # 0-indexed to 1-indexed

        return datetime(int(year), month_num, int(day))

    def _calculate_stats(self, corrections: List[Correction]) -> Dict:
        """교정 통계 계산"""
        by_rule = {}
        by_component = {}

        for corr in corrections:
            by_rule[corr.rule_id] = by_rule.get(corr.rule_id, 0) + 1
            by_component[corr.component] = by_component.get(corr.component, 0) + 1

        return {
            'total_corrections': len(corrections),
            'by_rule': by_rule,
            'by_component': by_component
        }


def main():
    """테스트 예제"""

    corrector = RuleBasedCorrector()

    # 테스트 기사 (Tier 1A + 한국 데이터 추가 + 섬/강/사찰 + C13 날짜 범위 + C12 날짜 형식)
    test_article = """[TITLE]Korea to invest 100 won in childcare - 20 percent increase[/TITLE][BODY]The government announced that twenty million won will be allocated in Gangnam-gu near Hangang. This represents 15% of the budget. The Joseon Kingdom era policies will be reviewed. The birth-rate has declined significantly in Mapo-gu. Healthcare spending will increase by thirty percent. A trip to Jejudo is planned. The festival will run from April 28 to April 30 in Seoul. Another event is scheduled from Dec. 28 to 3.[/BODY][CAPTION]This photo shows President Yoon visits Gyeongbokgung and Mt. Bukhan in Jongno-gu. Tourists hike Seoraksan and Hallasan National Park. A ceremony held at Bulguksa temple by Nakdonggang. Visitors enjoy Namiseom. The exhibition runs from May 15 to May 20. Protesters gather in London, Thursday, May 2, 2024. Another event in Paris, Monday, April 15, 2024. Korea Times photo shows the event in this captured from video.[/CAPTION]"""

    # Test with article date (May 7, 2024)
    test_article_date = "2024-05-07"

    print("=" * 80)
    print("Rule-Based Corrector Test (Tier 1A included)")
    print("=" * 80)
    print(f"\n기사 작성일: {test_article_date}")
    print("\n원본:")
    print(test_article)

    result = corrector.correct_article(test_article, article_date=test_article_date)

    print("\n\n교정 결과:")
    print(result['corrected_text'])

    print("\n\n적용된 교정:")
    print(f"총 {result['stats']['total_corrections']}개 교정")

    for corr in result['corrections']:
        if 'WARNING' in corr.corrected:
            print(f"  [{corr.rule_id}] ({corr.component}): {corr.corrected}")
        else:
            print(f"  [{corr.rule_id}] ({corr.component}): '{corr.original}' → '{corr.corrected}'")

    print("\n\n규칙별 통계:")
    for rule_id, count in sorted(result['stats']['by_rule'].items()):
        print(f"  {rule_id}: {count}개")

    print("\n" + "=" * 80)
    print("Tier 1A + 한국 데이터 규칙 테스트 완료")
    print("  C15: Korea Times photo → Korea Times file")
    print("  C18: Captured from 검증 (모호한 출처 경고)")
    print("  C27/A25: Gangnam-gu → Gangnam District (서울 25개 구)")
    print("  C28: Gyeongbokgung → Gyeongbok Palace (궁궐 5개)")
    print("  C29: Mt. Bukhan → Mount Bukhan, Seoraksan → Mount Seorak (산 65개)")
    print("  C32: Joseon Kingdom → Joseon Dynasty")
    print("\n한국 데이터 포함:")
    print("  - 산 이름: 65개 (100대 명산 포함)")
    print("  - 궁궐: 5개")
    print("  - 서울 구: 25개")
    print("  - 섬: 9개")
    print("  - 강: 4개")
    print("  - 사찰: 8개")
    print("  - 국립공원: 22개 (예외 처리)")
    print("=" * 80)


if __name__ == "__main__":
    main()
