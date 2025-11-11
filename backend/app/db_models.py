# Import all models once to ensure SQLAlchemy mapper registry is fully populated.
# This prevents late-binding issues for relationship("ClassName").

from .users.models import User  # noqa: F401
from .articles.models import Article, TextCorrectionHistory, ArticlePrompt  # noqa: F401
from .styleguides.models import StyleGuide, TextCorrectionHistoryStyle  # noqa: F401

