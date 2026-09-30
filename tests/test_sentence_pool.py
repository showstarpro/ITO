"""
Sanity checks for the three forward_sentence pooling modes: cls / mean / token
Usage (from the repo root):
    PYTHONPATH=src python tests/test_sentence_pool.py
    PYTHONPATH=src python tests/test_sentence_pool.py --model ViT-B-16 --device cuda
"""
import argparse

import torch

import open_clip


def run_model(model, img, text):
    out = model(img, text)
    if isinstance(out, dict):
        return out['sentence1_features'], out['sentence2_features']
    return out[3], out[4]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ViT-B-16')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    torch.manual_seed(0)

    tokenizer = open_clip.get_tokenizer(args.model)
    captions = ["a dog on grass", "a very long sentence about a cat sitting on a red sofa in the living room"]
    text = tokenizer(captions).to(args.device)
    eot = text.argmax(dim=-1)
    print('EOT positions:', eot.tolist())

    # two augmented views [2, B, 3, H, W]
    img = torch.randn(2, len(captions), 3, 224, 224, device=args.device)

    # replace padding after EOT with random tokens (id < EOT so argmax is unchanged) to check that padding is masked
    text_noisy = text.clone()
    pad = torch.arange(text.shape[1], device=args.device).unsqueeze(0) > eot.unsqueeze(-1)
    rand_ids = torch.randint(1, int(text.max()) - 1, text.shape, device=args.device)
    text_noisy[pad] = rand_ids[pad]
    assert torch.equal(text_noisy.argmax(dim=-1), eot)

    all_ok = True
    for pool in ['cls', 'mean', 'token']:
        print(f'\n===== sentence_pool = {pool} =====')
        model = open_clip.create_model(args.model, sentence_pool=pool).to(args.device)
        assert model.sentence_pool == pool
        print('has sentence_token:', hasattr(model, 'sentence_token'))

        # 1. forward: shape / NaN / normalization
        model.eval()
        with torch.no_grad():
            s1, s2 = run_model(model, img, text)
            s1_noisy, _ = run_model(model, img, text_noisy)
        print('shape:', tuple(s1.shape), '| nan:', torch.isnan(s1).any().item(), '| norm:', s1.norm(dim=-1).tolist())
        ok = s1.shape == (len(captions), model.text_projection.shape[-1] if not isinstance(model.text_projection, torch.nn.Linear) else model.text_projection.out_features)
        ok &= not torch.isnan(s1).any().item() and not torch.isnan(s2).any().item()
        ok &= torch.allclose(s1.norm(dim=-1), torch.ones_like(s1[:, 0]), atol=1e-4)

        # 2. padding must not affect the output
        diff = (s1 - s1_noisy).abs().max().item()
        print(f'padding invariance max diff: {diff:.2e}')
        ok &= diff < 1e-4

        # 3. the two views should give different outputs (different images)
        ok &= not torch.allclose(s1, s2)

        # 4. backward: sentence_transformer (and sentence_token) receive gradients
        model.train()
        s1, s2 = run_model(model, img, text)
        (s1.sum() + s2.sum()).backward()
        g = model.sentence_transformer.resblocks[0].attn.in_proj_weight.grad
        print('sentence_transformer grad norm:', None if g is None else g.norm().item())
        ok &= g is not None and torch.isfinite(g).all().item()
        if pool == 'token':
            g = model.sentence_token.grad
            print('sentence_token grad norm:', None if g is None else g.norm().item())
            ok &= g is not None and g.norm().item() > 0

        print('PASS' if ok else 'FAIL')
        all_ok &= bool(ok)

    print('\nALL PASS' if all_ok else '\nSOME FAILED')


if __name__ == '__main__':
    main()
