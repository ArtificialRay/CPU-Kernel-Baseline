"""Common base classes and type aliases for arm-bench data models."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonEmptyString = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]


class BaseModelWithDocstrings(BaseModel):
    """Base model that extracts attribute docstrings into the JSON schema."""

    model_config = ConfigDict(use_attribute_docstrings=True)
