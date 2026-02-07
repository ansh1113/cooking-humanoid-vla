# Dataset Documentation

## Overview
1,607 labeled video clips from 72 Indian cooking videos across 5+ chefs.

## Collection Methodology
1. **Video Download:** yt-dlp from YouTube
2. **Audio Segmentation:** Whisper (base model)
3. **Visual Features:** CLIP ViT-B/32
4. **Automated Labeling:** GPT-4V with audio context
5. **Verification:** 10% human validation (94% agreement)

## Action Vocabulary (40 Classes)
### Stirring Variants
- stirring curry, sautéing, simmering, frying onions, etc.

### Liquids
- adding water, adding oil, adding ghee, adding butter

### Solids
- adding vegetables, adding tomatoes, adding paneer, adding rice

### Spices
- adding masala, tempering spices, adding turmeric, garnishing

### Manipulation
- kneading dough, grinding paste, chopping vegetables

### Waiting
- pressure cooking, steaming, roasting

## Robot Primitives (8 Classes)
- STIR, POUR, TRANSFER, SPRINKLE, WAIT, PROCESS, PRESS, UNKNOWN

## Dataset Access
Full dataset available upon request for research purposes.
Contact: anshbhansali5@gmail.com
