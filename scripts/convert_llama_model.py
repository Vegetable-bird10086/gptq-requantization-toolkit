import torch
from collections import OrderedDict


def hf_to_et_key(key: str) -> str:
    # ===== 顶层 =====
    if key == "model.embed_tokens.weight":
        return "tok_embeddings.weight"
    if key == "model.norm.weight":
        return "norm.weight"
    if key == "lm_head.weight":
        return "output.weight"

    # ===== 层级 =====
    key = key.replace("model.layers.", "layers.")

    # ===== Attention =====
    key = key.replace(".self_attn.q_proj.weight", ".attention.wq.weight")
    key = key.replace(".self_attn.k_proj.weight", ".attention.wk.weight")
    key = key.replace(".self_attn.v_proj.weight", ".attention.wv.weight")
    key = key.replace(".self_attn.o_proj.weight", ".attention.wo.weight")

    # ===== FFN =====
    key = key.replace(".mlp.gate_proj.weight", ".feed_forward.w1.weight")
    key = key.replace(".mlp.up_proj.weight", ".feed_forward.w3.weight")
    key = key.replace(".mlp.down_proj.weight", ".feed_forward.w2.weight")

    # ===== Norm =====
    key = key.replace(".input_layernorm.weight", ".attention_norm.weight")
    key = key.replace(".post_attention_layernorm.weight", ".ffn_norm.weight")

    return key


def convert_state_dict(hf_state_dict):
    new_state_dict = OrderedDict()

    for k, v in hf_state_dict.items():
        new_k = hf_to_et_key(k)

        # 跳过不需要的（如果有）
        if "rotary_emb" in k:
            continue

        new_state_dict[new_k] = v

    return new_state_dict


def check_shapes(old_sd, new_sd):
    print("\n[INFO] Checking shapes...")
    for old_k, v in old_sd.items():
        new_k = hf_to_et_key(old_k)
        if new_k in new_sd:
            if v.shape != new_sd[new_k].shape:
                print(f"[WARNING] Shape mismatch:")
                print(f"  {old_k} {v.shape}")
                print(f"  {new_k} {new_sd[new_k].shape}")
    print("[INFO] Shape check done.\n")


def main():
    # ===== 输入路径 =====
    hf_model_path = "/root/autodl-tmp/llama3.2-1b-instruct-2bit/model.pth"   # 或 safetensors 转成的
    output_path = "/root/autodl-tmp/llama3.2-1b-instruct-2bit/model1.pth"

    print("[INFO] Loading HF model...")
    hf_state_dict = torch.load(hf_model_path, map_location="cpu")

    print("[INFO] Converting keys...")
    new_state_dict = convert_state_dict(hf_state_dict)

    check_shapes(hf_state_dict, new_state_dict)

    print("[INFO] Saving converted model...")
    torch.save(new_state_dict, output_path)

    print(f"[DONE] Saved to {output_path}")


if __name__ == "__main__":
    main()