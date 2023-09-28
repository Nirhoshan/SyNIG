# this entire fucntion is to analyse the pad point distribution of the actual traces.
# once an actual trace is selected, pad points are selected randomly from a distribution we created.

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import find_peaks


def extract_cross_section(im, r1, r2):
    im_cross_sec = im[r1:r2, :]
    return np.mean(im_cross_sec, axis=0)

def extract_cross_section_for_synth(im, r1, r2, win, slide):
    all_profiles = []
    start = r1
    while start < r2 - win:
        end = start + win
        all_profiles.append(extract_cross_section(im, start, end))
        # extract_cross_section_and_plot(im, start, end, start)

        start += slide

    return np.asarray(all_profiles)

'''analyse the zero padding regions on the image frames. Zero padding region right now, is fluctuating their pixel values.
which should be a constant pixel value'''
def get_zero_padding_point(trace_path):
    arr = pd.read_csv(trace_path, header=None).values

    profile = extract_cross_section(arr, r1=100, r2=125)

    # moving window. Look ahead for 10 samples. If the summaion of
    grad_profile = np.gradient(profile)
    w = 20
    for i in range(len(grad_profile) - w):
        is_all_zero = np.all(grad_profile[i:i + w] == 0)
        if is_all_zero:
            return i, profile

    return -1, profile


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


def reconstruct_given_img(image_profile, zero_pad_val, syn_pad_point):
    if syn_pad_point > 0:
        verti_arr = np.tile(image_profile, (125, 1))
        hori_arr = np.transpose(verti_arr)

        final_synth_arr = (verti_arr + hori_arr)
        final_synth_arr[:syn_pad_point, :] = final_synth_arr[:syn_pad_point, :] - zero_pad_val
        final_synth_arr[:, :syn_pad_point] = final_synth_arr[:, :syn_pad_point] - zero_pad_val
        final_synth_arr[:syn_pad_point, :syn_pad_point] = final_synth_arr[:syn_pad_point, :syn_pad_point] + zero_pad_val

        final_synth_arr[syn_pad_point:, syn_pad_point:] = final_synth_arr[syn_pad_point:, syn_pad_point:] / 2

        final_profile = extract_cross_section(final_synth_arr, r1=100, r2=125)

    else:
        verti_arr = np.tile(image_profile, (125, 1))
        plt.imshow(verti_arr)
        plt.show()
        hori_arr = np.transpose(verti_arr)
        final_synth_arr = (verti_arr + hori_arr) / 2

    return final_synth_arr

def sel_pad_point_for_synth(synth_profile, pad_point, act_profile, vid):

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

    synth_convolved_grad = np.gradient(synth_convolved)

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


def reconstruct_all_image(synth_profile, pad_point, act_profile, vid, trace, path_out):
    if pad_point > 0:

        syn_pad_point = sel_pad_point_for_synth(synth_profile, pad_point, act_profile, vid)
        zero_pad_val_act = act_profile[pad_point + 2]

        zero_pad_val_synth = np.mean(synth_profile[syn_pad_point:])
        weighted_zero_pad_val = 0.5 * zero_pad_val_act + 0.5 * zero_pad_val_synth

        synth_profile[syn_pad_point:] = weighted_zero_pad_val
        syn_image = reconstruct_given_img(synth_profile, weighted_zero_pad_val, syn_pad_point)

    else:
        syn_image = reconstruct_given_img(synth_profile, zero_pad_val=1, syn_pad_point=pad_point)
        syn_pad_point = -1

    df = pd.DataFrame(syn_image)
    assert len(df) > 0

    # store data
    df.to_csv(path_out, header=False, index=False)

    return syn_pad_point


def analyse_zero_padding_regions(video, platform, no_ori_traces, ori_path, gan_path, post_path):

    # process synthesized traces
    vid_synth_path_out = post_path + '/vid' + str(video + 1)
    if not os.path.exists(vid_synth_path_out):
        os.makedirs(vid_synth_path_out)

    num_of_actual_traces = no_ori_traces

    # get padding point details
    trace_profiles = []
    pad_points = []
    for t in range(num_of_actual_traces):
        print(' ' + str(t))
        # get zero padding point

        pad_p, profile = get_zero_padding_point(ori_path + '/deepfp_vid' + str(video + 1) + '_' + str(t + 1) + '.csv')
        trace_profiles.append(profile)
        pad_points.append([t + 1, pad_p])

    trace_profiles = np.asarray(trace_profiles)
    pad_points = np.asarray(pad_points)
    pad_lookup = pd.DataFrame(columns=['trace', 'act_pad_point'],
                              data=pad_points)

    selected_act_trace_for_least_mse = []
    synth_act_pads = []
    for s in range(10):
        print('vid ' + str(video + 1) + ' trace ' + str(s + 1))
        # synthesized trace profile
        synth_img = pd.read_csv(gan_path + '/vid' + str(video + 1) + '/deepfp_' + str(s + 1) + '.csv',
                                header=None).values

        synth_profiles = extract_cross_section_for_synth(synth_img,
                                                         r1=50,
                                                         r2=125,
                                                         win=25,
                                                         slide=10)

        # compare the synthe profiles with randomly selected traces
        rand_traces = np.random.choice(num_of_actual_traces,
                                       int(num_of_actual_traces * 0.2),
                                       replace=False)
        selected_act_profiles = trace_profiles[rand_traces, :]
        act_trace_ind_in_rand_trace, synth_sec_ind = get_mses(synth_profiles, selected_act_profiles)

        # get the pad points and reconstruct the image
        least_mse_act_profile = selected_act_profiles[act_trace_ind_in_rand_trace]
        least_mse_synth_profile = synth_profiles[synth_sec_ind]

        # extract the pad point
        least_mse_sel_act_trace = rand_traces[act_trace_ind_in_rand_trace]
        act_pad_point = pad_lookup[pad_lookup['trace'] == (least_mse_sel_act_trace + 1)].values[0, 1]

        syn_pad_point = reconstruct_all_image(least_mse_synth_profile, act_pad_point, least_mse_act_profile,
                                              video + 1,
                                              s + 1,
                                              vid_synth_path_out + '/deepfp_' + str(s + 1) + '.csv')
        selected_act_trace_for_least_mse.append(least_mse_sel_act_trace)

        df_sel_least_mse_trace = pd.DataFrame(columns=['trace'],
                                              data=np.asarray(selected_act_trace_for_least_mse).reshape([-1, 1]))

        # store data
        df_sel_least_mse_trace.to_csv(vid_synth_path_out + '/sel_least_mse_act_trace.csv', index=False)

        synth_act_pads.append([act_pad_point, syn_pad_point])

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

    ori_path = data_path + '/' + platform + '/traces_' + str(no_ori_traces) + '/actual/'
    gan_path = data_path + '/' + platform + '/traces_' + str(no_ori_traces) + '/gan/'
    post_path = data_path + '/' + platform + '/traces_' + str(no_ori_traces) + '/post/'

    analyse_zero_padding_regions(video, platform, no_ori_traces, ori_path, gan_path, post_path)



