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

    def _append(self, item: T) -> None:
        if "name" not in item.__dict__ and "_name" not in item.__dict__ and "name" not in type(item).__dict__:
            raise AttributeError("The appended item must have a name attribute")
        return super().append(item)

    def _remove(self, name: str) -> None:
        if name not in self.keys():
            raise AttributeError(f"The item named {name} cannot be removed because it it not in the list.")
        element_index = self.keys().index(name)
        return self.pop(element_index)

    def __getitem__(self, key: int | str) -> T:
        if isinstance(key, int):
            return super(NamedList, self).__getitem__(key)
        elif isinstance(key, str):
            for item in self:
                if item.name == key:
                    return item
            raise RuntimeError(f"The key {key} does not exist in the current object.")
        else:
            raise TypeError("key must be int or str")

    def __setitem__(self, key: int | str, value):
        if isinstance(key, int):
            return super(NamedList, self).__setitem__(key, value)
        elif isinstance(key, str):
            found_key = False
            for i, item in enumerate(self):
                if item.name == key:
                    self[i] = value
                    return
            if not found_key:
                return self.append(value)
        else:
            raise TypeError("key must be int or str")

    def __repr__(self):
        if not self:
            return "{}"

        items = []
        for element in self:
            # Get the name and the default repr
            name = element.name() if callable(element.name) else element.name
            default_repr = object.__repr__(element)
            items.append(f"{name} : {default_repr}")

        return "{\n " + ",\n ".join(items) + "\n}"

    def keys(self) -> list[str]:
        return [item.name for item in self]
