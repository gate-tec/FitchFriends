from typing import TypedDict, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Dict

__all__ = ["RelationDictionary", "WeightedRelationDictionary"]


RelationDictionary: TypedDict = TypedDict(
    'RelationDictionary',
    {0: list, 1: list, 'd': list}
)

WeightedRelationDictionary: TypedDict = TypedDict(
    'WeightedRelationDictionary',
    {0: "Dict[tuple, int]", 1: "Dict[tuple, int]", 'd': "Dict[tuple, int]"}
)
