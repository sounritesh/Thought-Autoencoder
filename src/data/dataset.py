from numpy import dtype
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# from src.utils.config import BASE_TOKENIZER


class Thoughts(Dataset):
    def __init__(self, ds, max_len, tokenizer) -> None:
        super().__init__()
        self.texts = ds.text.values
        self.tags = ds.tag.values
        self.user_bios = ds.bio.values
        self.top_level_flags = ds.is_top_level.values
        self.thought_blocked_flags = ds.is_thought_blocked.values
        self.anon_flags = ds.is_anon.values
        self.trigger_warning_flags = ds.has_trigger_warning.values
        self.profile_pic_flags = ds.has_profile_pic.values
        self.cover_image_flags = ds.has_cover_image.values
        self.pronoun_flags = ds.has_pronouns.values
        self.banned_flags = ds.is_banned_flags.values
        self.google_flags = ds.has_google
        self.phone_flags = ds.has_phone
        self.apple_flags = ds.has_apple
        self.location_flags = ds.has_location
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.banned_flags)

    def __getitem__(self, index):
        text = self.texts[index]
        user_bio =self.user_bios[index]

        tokens_text = self.tokenizer.encode_plus(
            text,
            max_length = self.max_len,
            padding = True,
            truncation = True,
            add_special_tokens = True
        )

        tokens_bio = self.tokenizer.encode_plus(
            user_bio,
            max_length = self.max_len/2,
            padding = True,
            truncation = True,
            add_special_tokens = True
        )

        return {
            'text_ids': torch.tensor(tokens_text['input_ids'], dtype=float),
            'text_mask': torch.tensor(tokens_text['attention_mask'], dtype=float),
            'text_type_ids': torch.tensor(tokens_text['token_type_ids'], dtype=float),
            'bio_ids': torch.tensor(tokens_bio['input_ids'], dtype=float),
            'bio_mask': torch.tensor(tokens_bio['attention_mask'], dtype=float),
            'bio_type_ids': torch.tensor(tokens_bio['token_type_ids'], dtype=float),
        }

class EmbeddingVector(Dataset):
    def __init__(self, text_encodings, bio_encodings, bio_ids, ds) -> None:
        super().__init__()
        self.inds = ds.index.values
        self.text_encodings = text_encodings
        self.bio_encodings = bio_encodings
        self.bio_text = ds.bio.values
        self.bio_unique = bio_ids.bio.values
        self.tags = OneHotEncoder(drop='if_binary').fit_transform(ds.tag.values.reshape(-1, 1))
        self.user_bios = ds.bio.values
        self.top_level_flags = ds.is_top_level.values
        self.thought_blocked_flags = ds.is_thought_blocked.values
        self.anon_flags = ds.is_anon.values
        self.trigger_warning_flags = ds.has_trigger_warning.values
        self.profile_pic_flags = ds.has_profile_pic.values
        self.cover_image_flags = ds.has_cover_image.values
        self.pronoun_flags = ds.has_pronouns.values
        self.banned_flags = ds.is_banned.values
        self.google_flags = ds.has_google.values
        self.phone_flags = ds.has_phone.values
        self.apple_flags = ds.has_apple.values
        self.location_flags = ds.has_location.values

    def __len__(self):
        return len(self.inds)

    def __getitem__(self, index):
        ind = self.inds[index]
        encoding = self.text_encodings[ind]
        bio_ind = np.where(self.bio_unique==self.bio_text[index])
        bio_encoding = self.bio_encodings[bio_ind]
        tag_encoding = self.tags[index].toarray().squeeze()
        other_encoding = np.array(
            [
                self.top_level_flags[index],
                self.thought_blocked_flags[index],
                self.anon_flags[index],
                self.trigger_warning_flags[index],
                self.profile_pic_flags[index],
                self.cover_image_flags[index],
                self.pronoun_flags[index],
                self.banned_flags[index],
                self.google_flags[index],
                self.phone_flags[index],
                self.apple_flags[index],
                self.location_flags[index]
            ]
        )

        print(encoding.shape, bio_encoding.shape, tag_encoding.shape, other_encoding.shape)
        return {
            "input_embed": torch.tensor(np.concatenate([encoding, bio_encoding, tag_encoding, other_encoding]))
        }