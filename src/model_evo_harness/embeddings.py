"""Validate declared field roles and shared embedding identities, without DL imports."""


def validate_embedding_roles(roles: object, fields: list[str]) -> list[dict]:
    """Check declarations; the host must verify actual modules and token mappings.

    Each role names raw fields, an encoding kind, output width and candidate module
    path. Categorical/sequence roles additionally bind ID domain and vocabulary.
    Optional encoding_transform identifies a mapping/hash/normalization version;
    sharing requires matching declarations. sequence_contract records max_length,
    mask_layout (default arbitrary), pooling (default none), weight_normalization
    (default none), and optional same-event alignment/length/mask/event-ID fields.
    Runtime utilities in native sequence_inputs verify masks and event alignment;
    these declarations do not prove actual data provenance.
    """
    if not isinstance(roles, list):
        raise ValueError("embedding_roles must be a list")
    known, seen, shared, aligned, result = set(fields), set(), {}, {}, []
    for role in roles:
        if not isinstance(role, dict):
            raise ValueError("each embedding role must be an object")
        for key in ("id", "module_path", "kind"):
            if not isinstance(role.get(key), str) or not role[key].strip():
                raise ValueError(f"embedding role needs {key}")
        if role["id"] in seen:
            raise ValueError("duplicate embedding role id")
        seen.add(role["id"])
        if role["kind"] not in ("numeric", "categorical", "sequence", "precomputed"):
            raise ValueError("invalid embedding role kind")
        inputs = role.get("fields")
        if (not isinstance(inputs, list) or not inputs or
                any(not isinstance(name, str) or name not in known for name in inputs) or
                len(set(inputs)) != len(inputs)):
            raise ValueError("embedding role fields must be unique task fields")
        width = role.get("width")
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            raise ValueError("embedding role width must be a positive integer")
        freeze = role.get("freeze_pretrained", False)
        if not isinstance(freeze, bool):
            raise ValueError("freeze_pretrained must be boolean")
        if "pretrained_digest" in role and (not isinstance(role["pretrained_digest"], str) or
                                            not role["pretrained_digest"].strip()):
            raise ValueError("pretrained_digest must identify the supplied table")
        if "encoding_transform" in role and (not isinstance(role["encoding_transform"], str) or
                                              not role["encoding_transform"].strip()):
            raise ValueError("encoding_transform must identify the exact mapping/transform")
        contract = role.get("sequence_contract")
        if "sequence_contract" in role:
            if role["kind"] != "sequence" or not isinstance(contract, dict):
                raise ValueError("sequence_contract belongs to a sequence role and must be an object")
            maximum = contract.get("max_length")
            if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < 1:
                raise ValueError("sequence_contract needs positive max_length")
            layout = contract.get("mask_layout", "arbitrary")
            if layout not in ("arbitrary", "prefix"):
                raise ValueError("mask_layout must be arbitrary or prefix")
            if contract.get("pooling", "none") not in ("none", "sum", "mean", "max"):
                raise ValueError("pooling must be none, sum, mean or max")
            if contract.get("weight_normalization", "none") not in ("none", "softmax"):
                raise ValueError("weight_normalization must be none or softmax")
            for key in ("mask_field", "length_field", "event_id_field"):
                if key in contract and (not isinstance(contract[key], str) or contract[key] not in known):
                    raise ValueError(f"{key} must identify a task field")
            group = contract.get("alignment_group")
            if "alignment_group" in contract:
                if not isinstance(group, str) or not group.strip():
                    raise ValueError("alignment_group must be a nonempty name")
                signature = (maximum, layout)
                if group in aligned and aligned[group] != signature:
                    raise ValueError("alignment group has inconsistent sequence contracts")
                aligned[group] = signature
        id_role = role["kind"] in ("categorical", "sequence")
        if id_role:
            for key in ("id_domain", "vocabulary_digest"):
                if not isinstance(role.get(key), str) or not role[key].strip():
                    raise ValueError(f"ID embedding role needs {key}")
            size = role.get("vocabulary_size")
            if not isinstance(size, int) or isinstance(size, bool) or size < 1:
                raise ValueError("vocabulary_size must be a positive integer")
            for key in ("padding_id", "oov_id"):
                value = role.get(key)
                if value is not None and (not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < size):
                    raise ValueError(f"{key} must be a vocabulary row or None")
            if role.get("padding_id") is not None and role.get("padding_id") == role.get("oov_id"):
                raise ValueError("padding and OOV must have distinct meanings")
        if role["kind"] == "precomputed":
            value = role.get("input_dim")
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError("precomputed role needs positive input_dim")
        signature = ("id" if id_role else role["kind"], width, role.get("id_domain"),
                     role.get("vocabulary_digest"), role.get("vocabulary_size"),
                     role.get("padding_id"), role.get("oov_id"), role.get("input_dim"),
                     role.get("pretrained_digest"), freeze, role.get("encoding_transform"))
        module = role["module_path"]
        if module in shared and shared[module] != signature:
            raise ValueError(f"shared embedding module {module} has incompatible role contracts")
        shared[module] = signature
        result.append({**role, "fields": list(inputs),
                       **({"sequence_contract": dict(contract)} if contract is not None else {})})
    return result
