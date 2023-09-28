# this entire fucntion is to analyse the pad point distribution of the actual traces.
# once an actual trace is selected, pad points are selected randomly from a distribution we created.

import pandas as pd
import os
from pyts.image import GramianAngularField
import numpy as np
from scipy.signal import find_peaks
import argparse


def extract_cross_section(im, r1, r2):
    im_cross_sec = im[r1:r2, :]
    return np.mean(im_cross_sec, axis=0)

'''
Extracting cross section for the synthetic images. 
'''
def extract_cross_section_for_synth(im, r1, r2, win, slide):
    all_profiles = []
    start = r1
    while start < r2 - win:
        end = start + win
        all_profiles.append(extract_cross_section(im, start, end))
        start += slide

    return np.asarray(all_profiles)


'''analyse the zero padding regions on the image frames. Zero padding region right now, is fluctuating their pixel values.
which should be a constant pixel value.'''
def get_zero_padding_point(trace_path):
    arr = pd.read_csv(trace_path, header=None).values

    profile = extract_cross_section(arr, r1=100, r2=125)
    # plot_profile_chang_ylim(profile, y_lim=[0, 1])

    # moving window. Look ahead for 10 samples. If the summaion of
    grad_profile = np.gradient(profile)
    grad_profile[grad_profile < 0.00000001] = 0

    # plot_profile_chang_ylim(grad_profile, y_lim=[-1, 1])

    w = 20

    for i in range(len(grad_profile) - w):
        is_all_zero = np.all(grad_profile[i:i + w] == 0)
        if is_all_zero:
            return i, profile, arr

    if i == (len(grad_profile) - w) - 1:
        i = -1

    return i, profile, arr


'''
Function providing the mse difference between the synthetic and actual traces
'''
def get_mses(synth_profiles, act_profiles):
    all_acts_mse = []
    for act_profile in act_profiles:
        all_secs_mse = []
        for sec_profile in synth_profiles:
            mse = ((act_profile - sec_profile) ** 2).mean()
            all_secs_mse.append(mse)
        all_acts_mse.append(all_secs_mse)

    all_acts_mse = np.asarray(all_acts_mse)

    ind_min = np.argwhere(all_acts_mse == np.min(all_acts_mse))

    return ind_min[0]

def mse(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def reconstruct_an_image_inverse_GASF_mapping(syn_profile, act_profile, act_image):
    syn_profile[syn_profile < 0] = 0
    act_profile[act_profile < 0] = 0

    trace_syn = np.sqrt((syn_profile + 1) / 2)
    trace_syn = 2 * trace_syn - 1

    trace_act = np.sqrt((act_profile + 1) / 2)
    trace_act = 2 * trace_act - 1

    a_max = np.max(trace_act)
    a_min = np.min(trace_act)

    s_max = np.max(trace_syn)
    s_min = np.min(trace_syn)

    m = (a_max - a_min) / (s_max - s_min)
    c = (s_max * a_min - s_min * a_max) / (s_max - s_min)
    syn_img_new = m * syn_profile + c
    syn_img_new = syn_img_new.reshape([1, -1])

    gasf = GramianAngularField(sample_range=(0, 1), method='summation')
    X_gasf = gasf.fit_transform(syn_img_new)[0]
    X_gasf = X_gasf * 0.5 + 0.5

    gamma_corrected_imgs = []
    ssim_test_vals = []
    for i, r in enumerate([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 1.0]):
        X_gasf_new = np.power(X_gasf, r)
        # score, diff = ssim(X_gasf_new, act_image, full=True)
        score = mse(X_gasf_new, act_image)

        ssim_test_vals.append(score)
        gamma_corrected_imgs.append(X_gasf_new)

    min_ind = np.argmin(np.asarray(ssim_test_vals))
    sel_im = gamma_corrected_imgs[min_ind]
    sel_im = 2 * sel_im - 1

    return sel_im


# select the
def sel_pad_point_for_synth(synth_profile, pad_point, act_profile, vid):
    # plot_profile(act_profile, rand_int=-1)
    # plot_profile(synth_profile, rand_int=-1)

    # smooth synth signal
    kernel_size = 10
    ma_start = 116
    kernel = np.ones(kernel_size) / kernel_size
    synth_convolved = np.convolve(synth_profile, kernel, mode='valid')

    val_list = []
    for i in range(ma_start, 125):
        val = 9 * synth_profile[i] + 8 * synth_profile[i - 1] + 7 * synth_profile[i - 2] + \
              6 * synth_profile[i - 3] + 5 * synth_profile[i - 4] + 4 * synth_profile[i - 5] + \
              3 * synth_profile[i - 6] + 2 * synth_profile[i - 7] + synth_profile[i - 8]
        val = val / 45
        val_list.append(val)
    synth_convolved = np.concatenate([synth_convolved, np.asarray(val_list)])
    # plot_profile(synth_convolved, rand_int=-1)

    synth_convolved_grad = np.gradient(synth_convolved)
    # plot_profile_chang_ylim(synth_convolved_grad, y_lim=[-0.1, 0.1])

    # run the pad point detection algorithm
    w = 10
    all_vals_mean = []
    start_point = pad_point - 5
    if start_point < 0:
        start_point = 0
    for i in range(start_point, 125 - w):
        pad_arr_grad_mean = np.mean(np.abs(synth_convolved_grad[i:i + w]))
        all_vals_mean.append(pad_arr_grad_mean)

    all_vals_mean = np.asarray(all_vals_mean)
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    all_vals_mean_convolved = np.convolve(all_vals_mean, kernel, mode='valid')

    # extract the first 2 lowest point and get  closest point to the given pad point.
    troughs, _ = find_peaks(-all_vals_mean_convolved)

    if len(troughs) > 0:
        first_3_points = troughs + start_point
        diff_with_pad = np.abs(first_3_points - pad_point)
        ind = first_3_points[np.argmin(diff_with_pad)]
    else:
        ind = pad_point

    return ind


def reconstruct_all_image(synth_profile, pad_point, act_profile, vid, trace, act_image, path_out):
    if pad_point > 0:

        syn_pad_point = sel_pad_point_for_synth(synth_profile, pad_point, act_profile, vid)

        zero_pad_val_act = act_profile[pad_point + 2]

        zero_pad_val_synth = np.mean(synth_profile[syn_pad_point:])
        weighted_zero_pad_val = 0.5 * zero_pad_val_act + 0.5 * zero_pad_val_synth

        synth_profile[syn_pad_point:] = weighted_zero_pad_val

        syn_image = reconstruct_an_image_inverse_GASF_mapping(synth_profile, act_profile, act_image)

    else:

        syn_image = reconstruct_an_image_inverse_GASF_mapping(synth_profile, act_profile, act_image)
        syn_pad_point = -1

    df = pd.DataFrame(syn_image)
    assert len(df) > 0

    # store data
    df.to_csv(path_out, header=False, index=False)

    return syn_pad_point


def analyse_zero_padding_regions(v, platform, original_traces, act_path, GAN_output_path, post_processed_path_out):

    # folder path to store the post processed synthetic video data
    # For each platform, we have multiple videos, therefore, we are crating one folder for each video to store
    # the data.
    activity_synth_path_out = post_processed_path_out + '/vid' + str(v + 1)
    if not os.path.exists(activity_synth_path_out):
        os.makedirs(activity_synth_path_out)

    num_of_actual_traces = original_traces

    # ++++++++++++++++++++++++++++++++++++
    # get 0 padding point details related to actual data. We capture the x or y axis position where the
    # 0 padding begins. We consider the slected number of actual traces (GASF images) from each video.
    trace_profiles = []
    pad_points = []
    trace_data = []
    for t in range(num_of_actual_traces):
        print(' ' + str(t + 1))
        pad_p, profile, arr = get_zero_padding_point(
            act_path + '/' + platform + '_vid' + str(v + 1) + '_' + str(t + 1) + '.csv')
        trace_profiles.append(profile)
        pad_points.append([t + 1, pad_p])
        trace_data.append(arr)

    trace_data = np.asarray(trace_data)
    trace_profiles = np.asarray(trace_profiles)
    pad_points = np.asarray(pad_points)
    pad_lookup = pd.DataFrame(columns=['trace', 'act_pad_point'],
                              data=pad_points)
    # store data for dubugging purposes
    pad_lookup.to_csv(activity_synth_path_out + '/padd_lookup.csv')
    # -------------------------------------

    # +++++++++++++++++++++++++++++++++++++
    # start the process for finding a suitable position for 0 padding if
    # the synthetic data in GASF format is elgible to have it
    selected_act_trace_for_least_mse = []
    synth_act_pads = []

    # number of synthetic data after GAN model to be post-processed.
    # this value can be changed based on the amount of synthetic data processed and
    # required amount of synthetic data to be processed.
    synth_ppost_proces_required = 2500
    for s in range(synth_ppost_proces_required):
        print('vidoe' + str(v + 1) + '_' + str(s + 1))

        # read synthetic image from the GAN model
        synth_img = pd.read_csv(GAN_output_path + '/vid' + str(v + 1) + '/' + platform + '_' + str(s + 1) + '.csv',
                                header=None).values

        # create cross sectional profiles for the synthetic image
        synth_profiles = extract_cross_section_for_synth(synth_img,
                                                         r1=50,
                                                         r2=125,
                                                         win=25,
                                                         slide=10)

        # compare the synthe profiles with randomly selected actual data
        rand_traces = np.random.choice(num_of_actual_traces,
                                       int(num_of_actual_traces * 0.2),
                                       replace=False)
        selected_act_profiles = trace_profiles[rand_traces, :]
        selected_act_trace = trace_data[rand_traces, :, :]

        # calculate the difference between the
        act_trace_ind_in_rand_trace, synth_sec_ind = get_mses(synth_profiles, selected_act_profiles)

        # get the pad points and reconstruct the image
        least_mse_act_profile = selected_act_profiles[act_trace_ind_in_rand_trace]
        least_mse_synth_profile = synth_profiles[synth_sec_ind]
        least_mse_act_image = selected_act_trace[act_trace_ind_in_rand_trace]

        # extract the pad point
        least_mse_sel_act_trace = rand_traces[act_trace_ind_in_rand_trace]
        act_pad_point = pad_lookup[pad_lookup['trace'] == (least_mse_sel_act_trace + 1)].values[0, 1]

        # reconstruct the synthetic post processed image.
        syn_pad_point = reconstruct_all_image(least_mse_synth_profile, act_pad_point, least_mse_act_profile,
                                              v + 1,
                                              s + 1,
                                              least_mse_act_image,
                                              activity_synth_path_out + '/' + platform + '_' + str(s + 1) + '.csv')
        selected_act_trace_for_least_mse.append(least_mse_sel_act_trace)

        df_sel_least_mse_trace = pd.DataFrame(columns=['trace'],
                                              data=np.asarray(selected_act_trace_for_least_mse).reshape([-1, 1]))

        # store data
        df_sel_least_mse_trace.to_csv(activity_synth_path_out + '/sel_least_mse_act_trace.csv', index=False)

        synth_act_pads.append([act_pad_point, syn_pad_point])

        break

    pad_lookup = pd.DataFrame(columns=['act_pad', 'syn_pad'],
                              data=np.asarray(synth_act_pads))

    # store data
    pad_lookup.to_csv(activity_synth_path_out + '/act_syn_pad_lookup.csv')

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--platform',
                        help='Enter the platforms either: YouTube, Netflix or Stan')
    parser.add_argument('--video',
                        help='Video ID ranging from 0 to 20',
                        type=int)
    parser.add_argument('--no_of_ori_traces',
                        help='Number of original traces for the algorithm',
                        type=int)
    parser.add_argument('--data_path',
                        help='Path to the original data folder')

    args = parser.parse_args()
    platform = args.platform
    video = args.video
    no_ori_traces = args.no_of_ori_traces
    data_path = args.data_path

    ori_path = data_path+'/'+ platform+'/traces_'+str(no_ori_traces)+'/actual/'
    gan_path = data_path+'/'+ platform+'/traces_'+str(no_ori_traces)+'/gan/'
    post_path = data_path+'/'+ platform+'/traces_'+str(no_ori_traces)+'/post/'

    analyse_zero_padding_regions(video, platform, no_ori_traces, ori_path, gan_path, post_path)


