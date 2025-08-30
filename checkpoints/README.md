# Model Checkpoints

## RS5M ViT-H-14 Checkpoint

The RS5M ViT-H-14 pre-trained model (3.8GB) is required for reproduction but not included in the GitLab submission due to size constraints.

### Download Instructions

```bash
# Download from official RS5M repository
wget https://github.com/om-ai-lab/RS5M/releases/download/v1.0/RS5M_ViT-H-14.pt

# Or use alternative download method
curl -L -o RS5M_ViT-H-14.pt https://github.com/om-ai-lab/RS5M/releases/download/v1.0/RS5M_ViT-H-14.pt

# Move to checkpoints directory
mv RS5M_ViT-H-14.pt checkpoints/
```

### Verification

The checkpoint should have:
- **Size**: ~3.8GB
- **SHA256**: (verify against official repository)
- **Architecture**: ViT-H-14 compatible with OpenCLIP

### Paper Reference

This checkpoint is used as described in:
> "We employed RS5M ViT-H-14, a vision transformer pre-trained on remote sensing imagery, with hierarchical prompt tuning"

**Citation**: Zhang et al. (2024) - RS5M: A large-scale vision-language dataset for remote sensing vision-language foundation model
