"""优化版回归验证；与正式实验使用相同的 KKT 容差。"""
from check_svm import main
from svm_optimized import SVMV1, SVMV2

if __name__ == "__main__":
    main(SVMV1, "v1_checks.json")
    main(SVMV2, "v2_checks.json")
