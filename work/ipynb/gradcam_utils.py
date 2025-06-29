
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def show_and_save_gradcam_examples(model, dataloader, class_labels, output_dir="gradcam_outputs", max_per_class=2):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    device = next(model.parameters()).device
    shown_correct = {label: 0 for label in class_labels}
    shown_wrong = {label: 0 for label in class_labels}
    model_name = type(model).__name__.lower()

    for batch_idx, (images, labels, filenames) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        for i, (img, label, pred, output, fname) in enumerate(zip(images, labels, preds, outputs, filenames)):
            img = img.unsqueeze(0).to(device)
            img.requires_grad_()
            actual = class_labels[label.item()]
            predicted = class_labels[pred.item()]
            correct = (label == pred)

            # if correct and shown_correct[actual] >= max_per_class:
            #     continue
            # if not correct and shown_wrong[actual] >= max_per_class:
            #     continue

            model.zero_grad()
            score = output[label.item()]
            score.backward(retain_graph=True)

            # Assume model has .feature_map_for_gradcam with registered hook
            gradients = model.feature_map_for_gradcam.grad.detach()
            activations = model.feature_map_for_gradcam.detach()

            pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
            for j in range(activations.shape[1]):
                activations[0, j, :, :] *= pooled_gradients[j]

            heatmap = torch.mean(activations, dim=1).squeeze()
            heatmap = F.relu(heatmap)
            heatmap /= torch.max(heatmap)
            heatmap = heatmap.cpu().numpy()

            img_np = img.squeeze(0).detach().cpu().numpy()
            img_np = img_np * np.array([0.229, 0.224, 0.225])[:, None, None] + np.array([0.485, 0.456, 0.406])[:, None, None]
            img_np = np.clip(img_np, 0, 1)
            img_np = np.transpose(img_np, (1, 2, 0))

            heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_color = np.float32(heatmap_color) / 255

            overlay = heatmap_color * 0.5 + img_np * 0.5
            overlay = np.clip(overlay, 0, 1)

            # 저장
            status = "correct" if correct else "wrong"
            filename = f"{model_name}{status}_{actual}_as_{predicted}_{fname}"
            save_fp = os.path.join(output_dir, filename)
            plt.imsave(save_fp, overlay)

            # 보여주기
            plt.figure(figsize=(5, 5))
            plt.imshow(overlay)
            plt.title(f"{'✔️' if correct else '❌'} Actual: {actual} | Pred: {predicted}")
            plt.axis('off')
            # plt.show()

            if correct:
                shown_correct[actual] += 1
            else:
                shown_wrong[actual] += 1

        # if all(shown_correct[k] >= max_per_class and shown_wrong[k] >= max_per_class for k in class_labels):
        #     break
