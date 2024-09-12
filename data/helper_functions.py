import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import json
import segyio
import skimage


def collate_fn(batch):
    max_traces_len = max(item["traces"].shape[0] for item in batch)
    max_bboxes_len = max(item["bboxes"].shape[0] for item in batch)

    for item in batch:
        traces_padding = max_traces_len - item["traces"].shape[0]
        if traces_padding > 0:
            item["traces"] = torch.nn.functional.pad(
                item["traces"], (0, 0, 0, traces_padding)
            )

        bboxes_padding = max_bboxes_len - item["bboxes"].shape[0]
        if bboxes_padding > 0:
            item["bboxes"] = torch.nn.functional.pad(
                item["bboxes"], (0, 0, 0, bboxes_padding)
            )

    traces = torch.stack([item["traces"] for item in batch])
    bboxes = torch.stack([item["bboxes"] for item in batch])

    return {"traces": traces, "bboxes": bboxes}


def plot_traces(traces, bboxes=None, true_centers=None, plot_path=None, logits=None):
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    if bboxes is not None:
        for trace1, depth1, trace2, depth2 in bboxes:
            box = patches.Rectangle(
                xy=(trace1, depth1),
                width=(trace2 - trace1),
                height=(depth2 - depth1),
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(box)

    if true_centers != None:
        for x, y in true_centers:
            xy = (float(x * traces.shape[0]), float(y * traces.shape[1]))
            circle = patches.Circle(xy, radius=5, color="blue")
            ax.add_patch(circle)

    if true_centers != None:
        for x, y in true_centers:
            xy = (float(x * traces.shape[0]), float(y * traces.shape[1]))
            circle = patches.Circle(xy, radius=5, color="blue")
            ax.add_patch(circle)

    if logits != None:
        for x, y in logits:
            xy = (float(x * traces.shape[0]), float(y * traces.shape[1]))
            circle = patches.Circle(xy, radius=3, color="yellow")
            ax.add_patch(circle)

    plt.xlabel("Trace")
    plt.ylabel("Samples")
    plt.imshow(traces[:, :].T, aspect="auto", cmap="gray")

    if plot_path is not None:
        plt.savefig(plot_path)


def plot_traces_classification_analysis(
    traces,
    bboxes=None,
    true_centers=None,
    plot_path=None,
    logits=None,
    preds_true_centers=None,
    preds_logits=None,
    each_file_avg=None,
):
    if each_file_avg != None:
        each_file_avg = [
            (int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, x2, y1, y2 in each_file_avg
        ]
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    if bboxes != None:
        for trace1, depth1, trace2, depth2 in bboxes:
            box = patches.Rectangle(
                xy=(trace1, depth1),
                width=(trace2 - trace1),
                height=(depth2 - depth1),
                linewidth=2,
                edgecolor="red",
                facecolor="none",
            )
            ax.add_patch(box)

    # if true_centers is not None and some == 1:
    #     for x, y in true_centers:
    #         xy = (float(x * traces.shape[0]), float(y * traces.shape[1]))
    #         circle = patches.Circle(xy, radius=5, color="blue")
    #         ax.add_patch(circle)

    if true_centers != None:
        for each_center in true_centers:
            x, y = each_center
            xy = (float(x * traces.shape[0]), float(y * traces.shape[1]))
            circle = patches.Circle(xy, radius=5, color="blue")
            ax.add_patch(circle)

    if logits != None:
        for x, y in logits:
            xy = (float(x * traces.shape[0]), float(y * traces.shape[1]))
            circle = patches.Circle(xy, radius=3, color="yellow")
            ax.add_patch(circle)

    if preds_true_centers != None:
        for i, each_pred in enumerate(preds_true_centers):
            if each_pred == 1.0:
                x, y = each_file_avg[i]
                xy = (float(x), float(y))
                circle = patches.Circle(xy, radius=5, color="yellow")
                ax.add_patch(circle)

    if preds_logits != None:
        for i, each_pred in enumerate(preds_logits):
            if each_pred == 1.0:
                x, y = each_file_avg[i]
                xy = (float(x), float(y))
                circle = patches.Circle(xy, radius=3, color="blue")
                ax.add_patch(circle)

    # ax.set_xticks(np.arange(0, traces.shape[0] + 1, 200))
    # ax.set_yticks(np.arange(0, traces.shape[1] + 1, 50))
    # ax.grid(True, which="both")

    plt.xlabel("Trace")
    plt.ylabel("Samples")
    plt.imshow(traces[:, :].T, aspect="auto", cmap="gray")

    if plot_path != None:
        plt.savefig(plot_path)


def save_pred(predictions, log_folder, ckpt_path, model_type):
    os.makedirs(os.path.dirname(f"{log_folder}/prediction.txt"), exist_ok=True)
    with open(f"{log_folder}/model.txt", "w") as f:
        f.write(f"model {ckpt_path=}")

    if model_type == "Regression":
        for j, batch_predictions in enumerate(predictions):
            traces, bboxes, true_centers, logits, file_name = batch_predictions
            for i in range(len(traces)):
                plot_path = log_folder + f"/pred_{j}_{i}.png"
                plot_traces(
                    traces[i], bboxes[i][0], true_centers[i], plot_path, logits[i]
                )

                np.save(
                    f"{log_folder}/true_centers_{j}_{i}.npy",
                    [each_true_center.numpy() for each_true_center in true_centers[i]],
                )
                np.save(
                    f"{log_folder}/logits_{j}_{i}.npy",
                    [each_logit.numpy() for each_logit in logits[i]],
                )
                np.save(
                    f"{log_folder}/bboxes_{j}_{i}.npy",
                    [each_bbox.numpy() for each_bbox in bboxes[i]],
                )
                np.save(
                    f"{log_folder}/traces_{j}_{i}.npy",
                    [each_trace.numpy() for each_trace in traces[i]],
                )

    if model_type == "Classification":
        a = 0
        for j, batch_predictions in enumerate(predictions):
            print(
                "----------------------ENTERED SAVE PRED-------------------------------------"
            )
            traces, target, logits, center, file_name = batch_predictions
            print(center)
            for i in range(len(traces)):
                plot_path = log_folder + f"/pred_{a}.png"
                plot_traces(traces[i], plot_path=plot_path)

                np.save(
                    f"{log_folder}/target_{a}.npy",
                    [each_true_center.numpy() for each_true_center in target[i]],
                )
                np.save(
                    f"{log_folder}/logits_{a}.npy",
                    [each_logit.numpy() for each_logit in logits[i]],
                )
                np.save(
                    f"{log_folder}/traces_{a}.npy",
                    [each_trace.numpy() for each_trace in traces[i]],
                )
                np.save(
                    f"{log_folder}/center_{a}.npy",
                    [each_trace.numpy() for each_trace in center[i]],
                )

                with open(f"{log_folder}/data_file.txt", "a") as file:
                    file.write(f"{a} - {file_name[i]}\n")

                a += 1


def open_segy_txt(segy_file, root_dir):
    base_name = os.path.splitext(segy_file)[0]
    annotation_file = base_name + ".txt"
    segy_path = os.path.join(root_dir, segy_file)
    traces = []
    with segyio.open(segy_path, "r", endian="little", strict=False) as segy_data:
        for i in range(segy_data.tracecount):
            traces.append(segy_data.trace[i])
        traces = np.array(traces)

    bboxes = []
    with open(os.path.join(root_dir, annotation_file), "r") as f:
        for line in f:
            if len(line) > 5:
                parts = line.strip().split(",")
                bbox = list(map(int, parts[:4]))
                bboxes.append(bbox)

    return traces, bboxes


def normalize_stacking(file_name, file_stacking_value=1, target_stacking_value=128):
    with segyio.open(file_name, mode="r+", endian="little", strict=False) as file:
        averaging_factor = target_stacking_value / file_stacking_value

        spec = segyio.tools.metadata(file)

        traces = []
        for i in range(len(file.trace)):
            traces.append(file.trace[i])
        traces = np.array(traces)

        print(traces.shape)
        original_trace_count = traces.shape[0]

        print(averaging_factor)

        if averaging_factor > 1:
            traces = np.transpose(
                skimage.measure.block_reduce(
                    np.transpose(traces), (1, int(averaging_factor)), np.mean
                )
            )
        if averaging_factor < 1:
            traces = np.transpose(
                np.repeat(np.transpose(traces), int((1 / averaging_factor)), axis=1)
            )
        if averaging_factor == 1:
            pass

        print(traces.shape)

        file.trace.raw[:] = traces[:]

        spec.tracecount = len(traces)

        with segyio.create(file_name, spec) as output_file:
            output_file.text[0] = file.text[0]

            for i in range(len(traces)):
                output_file.trace[i] = traces[i]
