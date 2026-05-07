"""Quick demo to validate drone video parsing and telemetry interpolation."""

from drone_parser import DroneVideoDataset


def main() -> None:
    dataset = DroneVideoDataset(
        "dataset",
        video_path="drone_footage/23-02-01_FR_F01_V01.MP4",
        csv_path="drone_footage/23-02-01_FR_F01.csv",
        target_fps=5.0,
        frame_stride=None,
    )
    print(dataset.summary())

    for i in range(5):
        frame, tele = dataset[i]
        print(
            f"[{i:02d}] frame={frame.shape} time={tele.time_s:.2f}s "
            f"pos_enu=({tele.pos_enu[0]:.2f}, {tele.pos_enu[1]:.2f}, {tele.pos_enu[2]:.2f})"
        )

    dataset.close()


if __name__ == "__main__":
    main()
