from typing import TypeVar, Generic

T = TypeVar("T")


class NamedList(list[T]):
    def __init__(self, *args, **kwargs):
        super(NamedList, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def append(self, item):
        if "name" not in item.__dict__:
            raise AttributeError("The appended item must have a name attribute")
        return super().append(object)

    def __getitem__(self, key):
        if isinstance(key, int):
            return super(NamedList, self).__getitem__(key)
        elif isinstance(key, str):
            for item in self:
                if item.name == key:
                    return item
        else:
            raise TypeError("key must be int or str")
