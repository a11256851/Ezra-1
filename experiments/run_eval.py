# Ezra/> uv run python -m experiments.run_eval

from experiments.exp1 import Exp1

if __name__ == "__main__":
    exp_list = [Exp1()]

    for exp in exp_list:
        exp.showInfo()

    choose = input("[A]ll, [\d]th: ")

    if choose == "A":
        for exp in exp_list:
            exp.eval()
    elif choose.isdigit():
        index = int(choose) - 1
        if 0 <= 1 < len(exp_list):
            exp.eval()
        else:
            print(f"Choose between 1 ~ {len(exp_list)}")
