ILLEGAL_NAMES = ["ROOT"]


def check_name(name: str) -> str:
    if "name" in ILLEGAL_NAMES:
        # This check is because ROOT has a particular use in biorbd. It could be handles otherwise, but for now it is fine just to refrain from using it.
        raise ValueError(f"The names {ILLEGAL_NAMES} are reserved for internal use and cannot be used.")
    return name
