Implementing the Transformer architecture from the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

Using this architecture to genrate Arabic Grapheme To Phoneme Conversion based on this paper [Transformer based Grapheme-to-Phoneme Conversion](https://arxiv.org/abs/2004.06338)

it also uses a Huge search space of 890K word to search for the word before using the model to save resources

[download weights](https://drive.google.com/file/d/19aP2ZO7QK-YfhjrotnR1Wx4ttJ_aE4g9/view?usp=drive_link) and add them to folder  **"dataset_weights"**

To use:

from G2P import Arabic_G2P

ex: Arabic_G2P('سلام')

ex: Arabic_G2P('هذا النص هو مثال')
