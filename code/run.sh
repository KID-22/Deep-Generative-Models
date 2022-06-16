nohup python ./main_gan.py --dataset=circle_1 >../log/gan_circle_1.log 2>&1

nohup python ./main_gan.py --dataset=circle_2 >../log/gan_circle_2.log 2>&1

nohup python ./main_wgan.py --dataset=circle_1 >../log/wgan_circle_1.log 2>&1

nohup python ./main_wgan.py --dataset=circle_2 >../log/wgan_circle_2.log 2>&1

nohup python ./main_wgan_gp.py --dataset=circle_1 >../log/wgan_gp_circle_1.log 2>&1

nohup python ./main_wgan_gp.py --dataset=circle_2 >../log/wgan_gp_circle_2.log 2>&1

nohup python ./main_vae.py --dataset=circle_1 >../log/vae_circle_1.log 2>&1

nohup python ./main_vae.py --dataset=circle_2 >../log/vae_circle_2.log 2>&1