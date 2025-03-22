from typing import TypeVar

T = TypeVar("T")


class NamedList(list[T]):
    def __init__(self, *args, **kwargs):
        super(NamedList, self).__init__(*args, **kwargs)

    @staticmethod
    def from_list(other: list[T]) -> "NamedList[T]":
        new_list = NamedList()
        for item in other:
            new_list.append(item)
        return new_list

    def append(self, item: T) -> None:
        if "name" not in item.__dict__ and "name" not in type(item).__dict__:
            raise AttributeError("The appended item must have a name attribute")
        return super().append(item)

    def __getitem__(self, key: int | str) -> T:
        if isinstance(key, int):
            return super(NamedList, self).__getitem__(key)
        elif isinstance(key, str):
            for item in self:
                if item.name == key:
                    return item
        else:
            raise TypeError("key must be int or str")

    def keys(self) -> list[str]:
        return [item.name for item in self]
