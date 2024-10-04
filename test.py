from utils_mt import *
from networks import *
root_path = os.getcwd()
print(root_path)

# folder to load config file
CONFIG_PATH = root_path
config = load_config(CONFIG_PATH, 'main_config.yaml')
#print('config: ', config)

exp = config['num_exp']
save_prob = config['save_prob_test']
times = config['times']
path_exp = root_path+'/' + config['folder_exp'] + '/exp'+str(exp)
path_models = path_exp+'/models'
path_maps = path_exp+'/pred_maps'

dir_data = config['data_directory']
tr_folder = 'train'
vl_folder = 'val'
channels = config['channels'] # each image
num_classes = config['num_classes']
num_domains = config['num_domains']

# list of folders to be merged
# SOURCE - TARGET1 - TARGET2
#list_dir = ['MT_1C', 'PA_1C', 'MA_1C']
list_dir = [config['sr_1'], config['tr_1']]
test_name = config['tr_1'][0:2]
patches_tg = 'patches_certain'


# parameters
lr = config['lr']
batch_size = config['batch_size']
patch_size = config['patch_size']
latent_dim = config['channels']
cdims = latent_dim//2
num_clases = config['num_classes']

lambda_r = config['lambda_r_']
lambda_ds = config['lambda_ds_']
lambda_de = config['lambda_de_']
#lambda_de = 0
lambda_c = config['lambda_c_']

print('Parameters')
print('\n', 'lr :', lr, '\n', 'batch_size: ', batch_size, '\n', 'patch_size: ', patch_size, '\n', 'num_clases: ', num_clases)

print('\n', 'lambda r :', lambda_r, '\n', 'lambda_ds: ', lambda_ds, '\n', 'lambda_de: ', lambda_de, '\n', 'lambda_c: ', lambda_c)

n_pool = 4
n_rows = 5
n_cols = 5
output_c_dim = 3
# overlap_percent = 0.4

im_test_dir = dir_data + test_name + '_1C/'
tr_img = np.load(im_test_dir + test_name + '_1C_RGB_img_filt_norm_2020_2021_10B.npy')
tr_row = tr_img.shape[0]
tr_col = tr_img.shape[1]
print('Test on: ', test_name + '_1C_RGB_img_filt_norm_2020_2021')
print('min, max values: ', np.min(tr_img), np.max(tr_img))
ref = np.load(im_test_dir + test_name + '_1C_ref_2020_2021.npy')

mask_test = np.load(im_test_dir +'test_mask_'+ test_name +'.npy')
print('test mask: ', 'test_mask_'+ test_name +'.npy')
ref = ref[:mask_test.shape[0], :mask_test.shape[1]]
print('clases: ', np.unique(ref))

# new size
patch_size_rows, patch_size_cols = new_shape_tiles(tr_img, n_pool, n_rows, n_cols)
print(patch_size_rows, patch_size_cols)
nb_filters_sh = 16
nb_filters_c = 16

if test_name == 'MA':
    pixel_a = 100
else:
    pixel_a = 625

print('pixel area: ', pixel_a)

if save_prob == False:
    prob_rec = np.zeros((times, mask_test.shape[0], mask_test.shape[1], 2), dtype=np.float32)

ts_time = []
for tm in range(0, times):
    print('time: ', tm)

    path_model_tm = path_models + '/run_' + str(tm)

    model_enc_sh = load_model(path_model_tm + '/' + 'enc_sh.h5', compile=False,
                              custom_objects={'InstanceNormalization': InstanceNormalization})
    model_classifier = load_model(path_model_tm + '/' + 'classifer.h5', compile=False)

    c = tr_img.shape[-1]
    print('shapes: ', tr_img.shape, ref.shape)
    new_model_enc_sh = build_encoder_sh((patch_size_rows, patch_size_cols, c), nb_filters=nb_filters_sh,
                                        name='enc_sh_new_')
    new_sh_out_shape = new_model_enc_sh.layers[-1].output_shape[1:]
    print('encoder sh output shape: ', new_sh_out_shape)
    new_model_classifier = build_classifier((new_sh_out_shape), nb_filters=nb_filters_c, num_classes=3, name='cl_new_')
    start_test = time.time()
    prob = inference_classifier_tiles(tr_img, n_pool, n_rows, n_cols, output_c_dim,
                                      model_enc_sh, new_model_enc_sh, model_classifier, new_model_classifier)
    elapsed_time = time.time() - start_test
    ts_time.append(elapsed_time)
    print('[****]', prob.shape)
    prob = prob[:mask_test.shape[0], :mask_test.shape[1], :]  ####
    print(prob.shape)
    print('prob range: ', np.min(prob[:, :, 1]), np.max(prob[:, :, 1]), prob.shape)
    # metrics
    #metrics_05, cm = matrics_AA_recall(ProbList, prob, ref, mask_test, pixel_a)
    #print('recall, precision, f1-score: ', metrics_05 * 100)
    #np.save(path_maps + '/' + 'confusion_matrix_' + str(tm) + '.npy', cm)

    if save_prob == True:
        print('[*****] Saving prob map ', tm)
        # np.save(path_maps + '/' + 'prob_map_' + list_dir[0][:2] + '_2_'+ test_name + '_' + str(tm) +'.npy', prob)
        np.savez_compressed(
            path_maps + '/' + 'prob_map_' + list_dir[0][:2] + '_2_' + test_name + '_' + str(tm) + '.npz',
            array_float32=prob)

    if save_prob == False:
        print('[*****] Prob map not saved ', tm)
        prob_rec[tm, :, :, :] = prob

    del prob, new_model_enc_sh, new_model_classifier
###
ts_time_ = np.asarray(ts_time)
np.save(path_exp + '/ts_times.npy', ts_time_)
del tr_img

if save_prob == True:
    prob_rec = np.zeros((times, mask_test.shape[0], mask_test.shape[1]), dtype = np.float32)
    for tm in range (0, times):
        print(tm)
        prob_rec[tm, :, :] = np.load(path_maps + '/' + 'prob_map_' + config['sr_1'][:2] + '_2_' +  config['tr_1'][:2] + '_' + test_name + '_' + str(tm) + '_fcn.npz')['array_float32']
        mean_prob = np.mean(prob_rec, axis=0)
else:
    mean_prob = np.mean(prob_rec[:, :, :, -1], axis=0)
print('average prob map:', mean_prob.shape)
ProbList = [0.5]
metrics_05, cm = matrics_AA_recall(ProbList, mean_prob, ref, mask_test, pixel_a)
print('average metrics \n recall, precision, f1-score: ', metrics_05 * 100)