import ast
from pathlib import Path


def test_best_checkpoint_is_reloaded_before_final_evaluation():
    source = Path("main_utils.py").read_text()
    tree = ast.parse(source)
    train_fn = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            isinstance(child, (ast.Constant, ast.Str))
            and "Reloaded best-primary model" in str(
                getattr(child, "value", getattr(child, "s", ""))
            )
            for child in ast.walk(node)
        )
    )
    calls = [
        node for node in ast.walk(train_fn) if isinstance(node, ast.Call)
    ]
    load_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "load_state_dict"
    ]
    final_eval_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Attribute)
        and node.func.attr == "evaluate_one_epoch"
        and any(
            isinstance(arg, ast.Name) and arg.id == "final_eval_epoch"
            for arg in node.args
        )
    ]
    assert load_calls
    assert final_eval_calls
    reload_call = min(load_calls, key=lambda call: call.lineno)
    assert isinstance(reload_call.func.value, ast.Name)
    assert reload_call.func.value.id == "model"
    assert min(call.lineno for call in load_calls) < min(
        call.lineno for call in final_eval_calls
    )
