import torch
import torch.nn as nn
import torch.nn.functional as F


from mmcv.ops import ModulatedDeformConv2d

from mmcv.runner import BaseModule, auto_fp16

from mmcv.cnn import ConvModule, xavier_init

from mmdet3d.models.builder import FUSION_LAYERS



@FUSION_LAYERS.register_module()
class BEVRSSMTemporalFusion(BaseModule):


    def __init__(
        self,

        in_channels,

        out_channels,

        kernel_size=3,

        deform_groups=1,

        latent_dim=None,

        hidden_dim=64,

        action_dim=2,

        kl_scale=1.0,


        norm_cfg=dict(
            type='BN',
            requires_grad=True
        ),

        act_cfg=dict(
            type='ReLU',
            inplace=True
        ),

        init_cfg=None
    ):


        super().__init__(init_cfg)


        if in_channels != out_channels:

            out_channels=in_channels


        if latent_dim is None:

            latent_dim=in_channels



        self.channels=out_channels

        self.latent_dim=latent_dim

        self.action_dim=action_dim

        self.kernel_size=kernel_size

        self.deform_groups=deform_groups

        self.padding=kernel_size//2

        self.kl_scale=kl_scale



        ################################################
        # deformable alignment
        ################################################


        self.offset_mask_generator=nn.Conv2d(

            in_channels*2,

            3*
            deform_groups*
            kernel_size*
            kernel_size,

            kernel_size,

            padding=self.padding
        )


        self.deform_conv=ModulatedDeformConv2d(

            in_channels,

            in_channels,

            kernel_size,

            padding=self.padding,

            deform_groups=deform_groups,

            bias=False
        )



        ################################################
        # Encoder
        ################################################


        self.encoder=nn.Sequential(

            ConvModule(

                in_channels,

                hidden_dim,

                3,

                padding=1,

                norm_cfg=norm_cfg,

                act_cfg=act_cfg
            ),


            ConvModule(

                hidden_dim,

                latent_dim,

                3,

                padding=1,

                norm_cfg=norm_cfg,

                act_cfg=None
            )

        )



        ################################################
        # transition h
        ################################################


        self.transition=ConvModule(

            in_channels+
            latent_dim+
            action_dim,

            out_channels,

            3,

            padding=1,

            norm_cfg=norm_cfg,

            act_cfg=act_cfg

        )



        ################################################
        # prior p(z|h)
        ################################################


        self.prior_mu=nn.Conv2d(

            out_channels,

            latent_dim,

            3,

            padding=1
        )


        self.prior_logstd=nn.Conv2d(

            out_channels,

            latent_dim,

            3,

            padding=1
        )



        ################################################
        # posterior q(z|h,e)
        ################################################


        self.posterior_mu=nn.Conv2d(

            out_channels+latent_dim,

            latent_dim,

            3,

            padding=1
        )


        self.posterior_logstd=nn.Conv2d(

            out_channels+latent_dim,

            latent_dim,

            3,

            padding=1
        )



        ################################################
        # decoder
        ################################################


        self.decoder=ConvModule(

            out_channels+latent_dim,

            out_channels,

            3,

            padding=1,

            norm_cfg=norm_cfg,

            act_cfg=None
        )



        ################################################
        # output
        ################################################


        self.output_layer=ConvModule(

            out_channels+latent_dim,

            out_channels,

            3,

            padding=1,

            norm_cfg=norm_cfg,

            act_cfg=act_cfg
        )


        self.init_weights()



    def init_weights(self):

        super().init_weights()


        nn.init.constant_(

            self.offset_mask_generator.weight,

            0
        )


        if self.offset_mask_generator.bias is not None:

            nn.init.constant_(

                self.offset_mask_generator.bias,

                0
            )


        xavier_init(

            self.deform_conv,

            distribution='uniform'
        )



    def sample(self,mu,logstd):


        std=torch.exp(logstd)


        eps=torch.randn_like(std)


        return mu+eps*std



    def kl_loss(

        self,

        mu_q,

        logstd_q,

        mu_p,

        logstd_p

    ):


        var_q=torch.exp(
            2*logstd_q
        )


        var_p=torch.exp(
            2*logstd_p
        )


        kl=(
            logstd_p-logstd_q

            +

            (
                var_q
                +(mu_q-mu_p)**2
            )
            /
            (
                2*var_p
            )

            -

            0.5

        )


        return kl.mean()



    @auto_fp16(
        apply_to=[
            'feat_curr',
            'feat_prev',
            'velocity'
        ]
    )
    def forward(

        self,

        feat_curr,

        feat_prev,

        velocity=None

    ):


        B,C,H,W=feat_curr.shape



        ################################################
        # deform alignment
        ################################################


        offset_mask=self.offset_mask_generator(

            torch.cat(
                [
                    feat_curr,
                    feat_prev
                ],
                dim=1
            )
        )


        k2=self.kernel_size*self.kernel_size


        offset_channels=(
            2*
            self.deform_groups*
            k2
        )


        offset=offset_mask[

            :,

            :offset_channels

        ]


        mask=offset_mask[

            :,

            offset_channels:

        ].sigmoid()



        h_prev=self.deform_conv(

            feat_prev,

            offset,

            mask
        )



        ################################################
        # velocity action
        ################################################


        if velocity is None:


            velocity=torch.zeros(

                B,

                self.action_dim,

                device=feat_curr.device
            )


        velocity=velocity[:,:,None,None]

        velocity=velocity.expand(

            B,

            self.action_dim,

            H,

            W
        )



        ################################################
        # prior previous z
        ################################################


        prior_mu_prev=self.prior_mu(h_prev)

        prior_std_prev=self.prior_logstd(h_prev)


        z_prev=self.sample(

            prior_mu_prev,

            prior_std_prev
        )



        ################################################
        # h transition
        ################################################


        h_t=self.transition(

            torch.cat(

                [
                    h_prev,

                    z_prev,

                    velocity
                ],

                dim=1
            )

        )



        ################################################
        # posterior
        ################################################


        e_t=self.encoder(

            feat_curr
        )


        mu_q=self.posterior_mu(

            torch.cat(

                [
                    h_t,

                    e_t
                ],

                dim=1
            )
        )


        logstd_q=self.posterior_logstd(

            torch.cat(

                [
                    h_t,

                    e_t
                ],

                dim=1
            )
        )



        z_t=self.sample(

            mu_q,

            logstd_q
        )



        ################################################
        # prior current
        ################################################


        mu_p=self.prior_mu(h_t)


        logstd_p=self.prior_logstd(h_t)



        ################################################
        # KL
        ################################################


        kl=self.kl_loss(

            mu_q,

            logstd_q,

            mu_p,

            logstd_p
        )



        ################################################
        # reconstruction
        ################################################


        reconstruction=self.decoder(

            torch.cat(

                [
                    h_t,

                    z_t
                ],

                dim=1
            )
        )



        ################################################
        # output
        ################################################


        output=self.output_layer(

            torch.cat(

                [
                    h_t,

                    z_t
                ],

                dim=1
            )
        )



        return (

            output,

            reconstruction,

            kl*self.kl_scale,

            h_t,

            z_t

        )