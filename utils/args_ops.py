from argparse import Action


class DictAction(Action):
    """
    argparse의 동작을 확장하여 인자(argument)를 KEY=VALUE 형태로 분할하고,
    첫 번째 '='을 기준으로 딕셔너리에 추가하는 동작(Action)입니다.
    리스트 옵션은 쉼표로 구분된 값으로 전달되어야 합니다. 예: KEY=V1,V2,V3
    Example:
        $ python3 ASC3/train.py \
        -d dataset.3asc.pickle \
        --run_name retraining \
        --random_forest \
            n_estimators=100 \
            n_jobs=30 \
            verbose=0 \
            class_weight="balanced"
        // in pyhton3
        >>> print(ARGS.random_forest)
        {
            n_estimators: 100,
            n_jobs: 30,
            verbose: 0,
            class_weight: "balanced",
        }
    """

    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ["true", "false"]:
            return True if val.lower() == "true" else False
        if val.lower() in ["none", "null"]:
            return None
        return val

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split("=", maxsplit=1)
            val = [self._parse_int_float_bool(v) for v in val.split(",")]
            if len(val) == 1:
                val = val[0]
            options[key] = val
        setattr(namespace, self.dest, options)
