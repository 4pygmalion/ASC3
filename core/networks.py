from typing import Tuple, List
from sparsemax import Sparsemax

import torch
import torch.nn as nn


AVAILABLE_MODELS = {
    "snv_model": {
        "AttentionMIL",
        "TestAugAttentionMIL",
        "ContributionMomentAttentionMIL",
    },
    "multimodal_model": {
        "MultimodalAttentionMIL",
        "TabnetEncoderMAMIL",
    },
}


class AttentionLayer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super(AttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        alignment = self.linear(x).squeeze(dim=-1)  # (n, 1) -> (n, )
        attention_weight = torch.softmax(alignment, dim=0)  # (n,)
        return attention_weight


class GatedAttentionLayer(nn.Module):
    """Gated attetention layer module.

    See Also:
        - Ilse, M., Tomczak, J., & Welling, M. (2018, July),
        Attention-based deep multiple instance learning.
        In International conference on machine learning (pp. 2127-2136). PMLR.

    """

    def __init__(self, input_dim: int, hidden_dim=10, use_sparsemax=False) -> None:
        super(GatedAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.v = nn.Linear(input_dim, hidden_dim, bias=False)
        self.u = nn.Linear(input_dim, hidden_dim, bias=False)
        self.w = nn.Linear(hidden_dim, 1, bias=False)
        self.softmax = Sparsemax(dim=1) if use_sparsemax else torch.nn.Softmax(dim=1)

    def forward(self, x):
        vh = torch.tanh(self.v(x))
        uh = torch.sigmoid(self.u(x))
        attention_score = self.w(vh * uh).squeeze(dim=-1)  # (n, 1) -> (n, )
        attention_weight = self.softmax(attention_score.unsqueeze(0)).squeeze(
            0
        )  # (n,) -> (1,n)-> softmax(1,n)-> (n,)
        return attention_weight


class AttentionMIL(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_hiddens: int,
        n_att_hiddens: int = 10,
    ):
        super(AttentionMIL, self).__init__()
        self.n_features = n_features
        self.n_hiddens = n_hiddens

        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, n_hiddens),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.attention_layer = GatedAttentionLayer(input_dim=n_hiddens, hidden_dim=n_att_hiddens)
        self.classifier = nn.Linear(n_hiddens, 1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.squeeze(dim=0)  # (1, instance, features) -> (instance, feature)

        instance_features = self.encoder(x)
        attention_weights = self.attention_layer(instance_features)
        weighted_features = torch.einsum("i,ij->ij", attention_weights, instance_features)

        context_vector = weighted_features.sum(axis=0)
        logit = self.classifier(context_vector)
        instance_contribution = attention_weights * self.classifier(instance_features).squeeze(
            1
        )  # (n, 1)

        return (logit, instance_contribution)


class TestAugAttentionMIL(AttentionMIL):
    generator = torch.random.manual_seed(0)

    def __init__(self, *args, **kargs):
        super(TestAugAttentionMIL, self).__init__(*args, **kargs)

    def __get_prob(self, x):
        instance_features = self.encoder(x)
        attention_weights = self.attention_layer(instance_features)
        weighted_features = torch.einsum("i,ij->ij", attention_weights, instance_features)
        instance_contributions = self.classifier(weighted_features)

        context_vector = weighted_features.sum(axis=0)
        logit = self.classifier(context_vector)

        return logit, instance_contributions

    def forward(self, x):
        """

        Note:
            - input tensor에 gussian noise 추가하여 inference 결과에 대한 분포 생성
            - 이후 mean 값으로 y_hat을 추정

        """
        if x.ndim == 3:
            x = x.squeeze(dim=0)  # (1, instance, features) -> (instance, feature)

        if self.training:
            logit, instance_contributions = self.__get_prob(x)

        else:
            n_augmentation = 5
            noise_factor = 0.1
            logit = 0.0
            instance_contributions = 0.0

            device = self.parameters().__next__().device.type
            noise = torch.normal(
                0, 1, (n_augmentation, len(x), self.n_features), generator=self.generator
            ).to(device)

            for idx in range(n_augmentation):
                inp = x + noise[idx] * noise_factor
                logit_sample, instance_contribution_sample = self.__get_prob(inp)
                logit += logit_sample
                instance_contributions += instance_contribution_sample

            logit /= 5
            instance_contributions /= 5

        return (logit, instance_contributions.squeeze(dim=0))


class ContributionMomentAttentionMIL(AttentionMIL):
    def __init__(self, *args, **kargs):
        super(ContributionMomentAttentionMIL, self).__init__(*args, **kargs)
        self.moments = ["mean", "std", "max", "min"]
        self.classifier = nn.Linear(self.n_hiddens + len(self.moments), 1)

    def forward(self, x):
        if x.ndim == 3:
            x = x.squeeze(dim=0)  # (1, instance, features) -> (instance, feature)

        instance_features = self.encoder(x)
        attention_weights = self.attention_layer(instance_features)
        weighted_features = torch.einsum("i,ij->ij", attention_weights, instance_features)
        instance_contributions = weighted_features.sum(axis=-1)

        context_vector = weighted_features.sum(axis=0)
        moment_features = [
            getattr(instance_contributions, moment)().unsqueeze(0) for moment in self.moments
        ]
        logit = self.classifier(torch.cat([context_vector] + moment_features))

        return (logit, instance_contributions.squeeze(dim=0))


class MultimodalAttentionMIL(nn.Module):
    def __init__(
        self,
        n_snv_features: int,
        n_cnv_features: int,
        n_hiddens: int,
        n_att_hiddens: int = 10,
        use_sparsemax: bool = False,
    ):
        super(MultimodalAttentionMIL, self).__init__()
        self.n_snv_features = n_snv_features
        self.n_cnv_features = n_cnv_features
        self.n_hiddens = n_hiddens

        self.encoder_snv = nn.Sequential(
            nn.Linear(n_snv_features, 32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32, n_hiddens),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.encoder_cnv = nn.Sequential(
            nn.Linear(n_cnv_features, 16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(16, n_hiddens),
            nn.Dropout(0.2),
            nn.ReLU(),
        )
        self.encoded_vec_projection = nn.Linear(n_hiddens, n_hiddens)
        self.attention_layer = GatedAttentionLayer(
            input_dim=n_hiddens, hidden_dim=n_att_hiddens, use_sparsemax=use_sparsemax
        )
        self.classifier = nn.Linear(n_hiddens, 1)
        self.instance_classifier = nn.Linear(n_hiddens, 1)

    def forward(self, x):
        snv_x, cnv_x = x

        if snv_x.ndim == 3:
            snv_x = snv_x.squeeze(dim=0)
        if cnv_x.ndim == 3:
            cnv_x = cnv_x.squeeze(dim=0)

        snv_encoded = self.encoder_snv(snv_x)
        cnv_encoded = self.encoder_cnv(cnv_x)
        instance_features = self.encoded_vec_projection(
            torch.cat([snv_encoded, cnv_encoded], axis=0)
        )

        attention_weights = self.attention_layer(instance_features)
        weighted_features = torch.einsum("i,ij->ij", attention_weights, instance_features)

        context_vector = weighted_features.sum(axis=0)
        logit = self.classifier(context_vector)
        instance_contribution = attention_weights * (
            self.instance_classifier(instance_features).squeeze(dim=1)
        )  # self.classifier(instance_features).squeeze(dim=1)
        return (logit, instance_contribution)


class TabNetVariantEncoder(nn.Module):
    def __init__(self, tabnet_encoder: nn.Module, output_dim=128) -> None:
        """TabNet variant encoder

        Args:
            tabnet_encoder (nn.Module): TabNetPretraining.network.encoder
            output_dim (int, optional): output feature dimension. Defaults to 128.

        Note:
            - TabNetPretraining 내에 다음과 같은 속성의 torch.nn.module이 존재
                .TabNetEncoder
                .EmbeddingGenerator
                .TabNetDecoder

            TabNetEncoder: 임베딩이 제외된 TabNet인코딩 파트
            EmbeddingGenerator: 입력feature중에 categorical 변수가 있어
                TabNetPretraining중에 cat_ 관련 인자를 전달하는 경우 생성
                인자가 없는 경우x -> x가 나옴
            TabNetDecoder: 디코더 파트

            eval모드:
                TabNetPretraining가 foward중에 카테고리컬 변수가 있으면 카테고리컬변수를
                인코딩한 벡터를 concat한 x가 전달되어, encoder에 전달됨
                카테고리별 변수가 없으면 x->x가 전달되어 encoder에 전달.


        """
        super().__init__()
        self.encoder = tabnet_encoder
        self.emb = torch.nn.Linear(self.encoder.n_d, output_dim)  # n_d: dicision vector dim

    def forward(self, x):
        steps_output, m_loss = self.encoder(x)
        x = torch.stack(steps_output, dim=0).mean(dim=0)  # List[torch.Tensor] -> torch.Tensor
        x = self.emb(x)

        return x


class TabNetMIL(torch.nn.Module):
    def __init__(
        self,
        snv_encoder: nn.Module,
        cnv_encoder: nn.Module,
        tabnet_output_dim: int = 128,
        hidden_dim: int = 64,
    ):
        """
        Example:
            >>> snv_tabnet = TabNetPretrainer()
            >>> cnv_tabnet = TabNetPretrainer()

            >>> tabnet_train_config = {
                    "max_epochs":100,
                    "patience":5,
                    "batch_size":2048,
                    "virtual_batch_size":128,
                    "pretraining_ratio": 0.5  # 재추축을 위해 마스킹 하는 비율
                }
            >>> snv_tabnet.fit(
                    train_data.snv_x.astype(np.float32),
                    [val_data.snv_x.astype(np.float32)],
                    **tabnet_train_config
                )
            >>> cnv_tabnet.fit(
                    train_data.cnv_x.astype(np.float32),
                    [val_data.cnv_x.astype(np.float32)],
                    **tabnet_train_config
                )
            >>> tabnet_mil = TabNetMIL(snv_tabnet.network, cnv_tabnet.network)
        """
        super(TabNetMIL, self).__init__()
        self.snv_featurizer = TabNetVariantEncoder(snv_encoder, tabnet_output_dim)
        self.cnv_featurizer = TabNetVariantEncoder(cnv_encoder, tabnet_output_dim)
        self.shared_embedding = torch.nn.Linear(tabnet_output_dim, hidden_dim)
        self.attention_layer = GatedAttentionLayer(hidden_dim, use_sparsemax=True)
        self.classifier = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        snv, cnv = x
        if snv.ndim == 3:
            snv = snv.squeeze(dim=0)
        if cnv.ndim == 3:
            cnv = cnv.squeeze(dim=0)

        featurized_x = self.snv_featurizer(snv)

        # tabnet의 ghost norm 때문에 instance 개수가 적어도 2개 이상이어야함
        if len(cnv) > 1:
            featurized_cnv_x = self.cnv_featurizer(cnv)
            featurized_x = torch.concat([featurized_x, featurized_cnv_x], dim=0)

        emb_v = self.shared_embedding(featurized_x)

        attention_weight = self.attention_layer(emb_v)  # (n,)
        weighted_h = torch.einsum("i,ij->ij", attention_weight, emb_v)  # (n, emb)
        context_vector = weighted_h.sum(axis=0)  # (emb,)
        bag_logit = self.classifier(context_vector)

        instance_contributions = attention_weight * self.classifier(emb_v).squeeze(dim=1)

        return (bag_logit, instance_contributions)
