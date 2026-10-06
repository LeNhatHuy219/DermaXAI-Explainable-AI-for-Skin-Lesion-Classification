# Launcher intervention nhiều seeds cho M4-ST, mặc định 42/123/2026.
# Đầu vào: frozen checkpoints/test exports ST và annotation concepts của ca test.
# main() gọi m4_interventions_common.run_cli("straight_through"); không train model.
# Đầu ra: results/<metric>/m4_st/<cấu hình>/intervention/, hoặc --output_dir.
# Gồm JSON, CSV, report.md và biểu đồ; config_tag phân biệt head/LR/epochs.

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.m4_interventions_common import run_cli


# Gọi pipeline intervention nhiều seeds và báo cáo chung, cố định variant ST.
def main(argv=None):
    return run_cli("straight_through", argv)


if __name__ == "__main__":
    main()
