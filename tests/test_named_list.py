import pytest
from biobuddy.utils.named_list import NamedList


class SimpleNamedObject:
    def __init__(self, name):
        self.name = name


class PropertyNamedObject:
    def __init__(self, name_value):
        self._name = name_value

    @property
    def name(self):
        return self._name


def test_init_empty():
    """Test initialization of an empty NamedList."""
    named_list = NamedList()
    assert len(named_list) == 0
    assert named_list.keys() == []


def test_init_with_items():
    """Test initialization with items."""
    items = [SimpleNamedObject("item1"), SimpleNamedObject("item2")]
    named_list = NamedList(items)
    assert len(named_list) == 2
    assert named_list.keys() == ["item1", "item2"]


def test_from_list():
    """Test creating a NamedList from a regular list."""
    regular_list = [SimpleNamedObject("item1"), SimpleNamedObject("item2")]
    named_list = NamedList.from_list(regular_list)
    assert len(named_list) == 2
    assert named_list.keys() == ["item1", "item2"]
    assert isinstance(named_list, NamedList)


def test_append():
    """Test appending items to a NamedList."""
    named_list = NamedList()
    named_list._append(SimpleNamedObject("item1"))
    assert len(named_list) == 1
    assert named_list.keys() == ["item1"]

    # Append another item
    named_list._append(SimpleNamedObject("item2"))
    assert len(named_list) == 2
    assert named_list.keys() == ["item1", "item2"]


def test_append_no_name_attribute():
    """Test appending an item without a name attribute raises an error."""

    class NoNameObject:
        pass

    named_list = NamedList()
    with pytest.raises(AttributeError, match="must have a name attribute"):
        named_list._append(NoNameObject())


def test_append_with_property_name():
    """Test appending an item with a property name."""
    named_list = NamedList()
    named_list._append(PropertyNamedObject("prop_item"))
    assert len(named_list) == 1
    assert named_list.keys() == ["prop_item"]


def test_remove():
    """Test removing items from a NamedList."""
    named_list = NamedList()
    named_list._append(SimpleNamedObject("item1"))
    named_list._append(SimpleNamedObject("item2"))

    removed = named_list._remove("item1")
    assert len(named_list) == 1
    assert named_list.keys() == ["item2"]
    assert removed.name == "item1"


def test_remove_nonexistent():
    """Test removing a nonexistent item raises an error."""
    named_list = NamedList()
    named_list._append(SimpleNamedObject("item1"))

    with pytest.raises(AttributeError, match="cannot be removed because it it not in the list"):
        named_list._remove("nonexistent")


def test_getitem_by_index():
    """Test getting an item by index."""
    named_list = NamedList()
    item1 = SimpleNamedObject("item1")
    item2 = SimpleNamedObject("item2")
    named_list._append(item1)
    named_list._append(item2)

    assert named_list[0] is item1
    assert named_list[1] is item2


def test_getitem_by_name():
    """Test getting an item by name."""
    named_list = NamedList()
    item1 = SimpleNamedObject("item1")
    item2 = SimpleNamedObject("item2")
    named_list._append(item1)
    named_list._append(item2)

    assert named_list["item1"] is item1
    assert named_list["item2"] is item2


def test_getitem_nonexistent_name():
    """Test getting a nonexistent item by name raises an error."""
    named_list = NamedList()
    named_list._append(SimpleNamedObject("item1"))

    with pytest.raises(RuntimeError, match="does not exist in the current object"):
        _ = named_list["nonexistent"]


def test_getitem_invalid_key_type():
    """Test getting an item with an invalid key type raises an error."""
    named_list = NamedList()

    with pytest.raises(TypeError, match="key must be int or str"):
        _ = named_list[1.5]


def test_setitem_by_index():
    """Test setting an item by index."""
    named_list = NamedList()
    item1 = SimpleNamedObject("item1")
    item2 = SimpleNamedObject("item2")
    named_list._append(item1)

    named_list[0] = item2
    assert named_list[0] is item2
    assert named_list.keys() == ["item2"]


def test_setitem_by_name_existing():
    """Test setting an existing item by name."""
    named_list = NamedList()
    item1 = SimpleNamedObject("item1")
    item2 = SimpleNamedObject("item2")
    named_list._append(item1)

    named_list["item1"] = item2
    assert named_list[0] is item2
    assert named_list.keys() == ["item2"]


def test_setitem_by_name_new():
    """Test setting a new item by name."""
    named_list = NamedList()
    item1 = SimpleNamedObject("item1")

    named_list["new_key"] = item1
    assert len(named_list) == 1
    assert named_list[0] is item1
    assert named_list.keys() == ["item1"]


def test_setitem_invalid_key_type():
    """Test setting an item with an invalid key type raises an error."""
    named_list = NamedList()

    with pytest.raises(TypeError, match="key must be int or str"):
        named_list[1.5] = SimpleNamedObject("item1")


def test_repr_empty():
    """Test string representation of an empty NamedList."""
    named_list = NamedList()
    assert repr(named_list) == "{}"


def test_repr_with_items():
    """Test string representation of a NamedList with items."""
    named_list = NamedList()
    named_list._append(SimpleNamedObject("item1"))
    named_list._append(SimpleNamedObject("item2"))

    repr_str = repr(named_list)
    assert "item1 :" in repr_str
    assert "item2 :" in repr_str
    assert repr_str.startswith("{\n")
    assert repr_str.endswith("\n}")
    assert "test_named_list.SimpleNamedObject object at" in repr_str


def test_keys():
    """Test getting the keys of a NamedList."""
    named_list = NamedList()
    named_list._append(SimpleNamedObject("item1"))
    named_list._append(SimpleNamedObject("item2"))

    assert named_list.keys() == ["item1", "item2"]
