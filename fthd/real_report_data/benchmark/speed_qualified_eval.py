import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
import re
import torch
from models_supervised import string_to_dataset, string_to_model
# from train_supervised import train
from train_finetuning import train

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
weights = torch.tensor([1.0, 1.0, 1.0]).to(device)
zoom_time = 0.0
zoom_vx_time = 0.0
zoom_vx_lower = 0.0

zoom_vx_upper = 0.0
    

dir_name = os.path.dirname(__file__)

fontsize = 26
tick_label_fontsize = 26
legend_fontsize = 21

plot_linewidth = 3

plot_original = True
color = ["red","black","blue","brown"]

color = {
    "5":"green",
    "15":"red",
    "30":"black",
    "60":"blue",
    "90":"brown"
}

linestyle = {
    "5":"--",
    "15":"--",
    "30":"--",
    "60":"-.",
    "90":":"
}
training_percent = ["90","60","30","15","5"]
ground_truth = False



# denoised_features = denoised_features.squeeze(1)
# print("denoised_feature shape:",denoised_features.shape)
if plot_original:
    data_file = os.path.join(dir_name,"data/Putnam_park2023_run4_2_1RNN_Val.npz")

    model_folder = os.path.join(dir_name,"data/ddm_model")
else:
    data_file = os.path.join(dir_name,"data/denoised_csv_1RNN_val.npz")
    # data_file = os.path.join(dir_name,"data/Putnam_park2023_run4_2_1RNN_Val.npz")

    model_folder = os.path.join(dir_name,"data/ekf_fthd_model")
model_files = os.listdir(model_folder)

pth_files = [f for f in model_files if f.endswith('.pth')]

patterns = {
    'layers': r'(\d+)layers',
    'neurons': r'(\d+)neurons',
    'batch': r'(\d+)batch',
    'lr': r'(\d+\.\d+)lr',
    'horizon': r'(\d+)horizon',
    'gru': r'(\d+)gru',
    'p': r'(\d+)p'
}

fig, axs = plt.subplots(4, 1, figsize=(10, 25))

# Define the function to convert the list of lists of arrays
def convert_list_to_array_correct(list_of_arrays):
    # Transpose the list of lists to get the desired order
    # transposed_list = list(map(list, zip(*list_of_arrays)))
    # Concatenate the arrays within each sublist along the first axis
    final_array = np.vstack(list_of_arrays)
    # Stack the resulting arrays along the first axis to get a single array
    # final_array = np.vstack(concatenated_sublist)
    return final_array

for j,filename in enumerate(pth_files):
    
    filepath = os.path.join(model_folder, filename)
    # Load the file (assuming you are using torch to load .pth files)
    # model = torch.load(filepath) # Uncomment and customize this line based on your specific use case
    print(f"Loaded file: {filename}")
    


    # Extracting the numbers
    extracted_values = {key: re.search(pattern, filename).group(1) for key, pattern in patterns.items()}

    print(extracted_values['p'])
    if plot_original:
        model_cfg = "deep_dynamics_iac.yaml"
    else:
        model_cfg = "fthd.yaml"

    model_cfg = os.path.join(dir_name,"model_cfg",model_cfg)

    with open(model_cfg, 'rb') as f:
        param_dict = yaml.load(f, Loader=yaml.SafeLoader)
        
    param_dict["MODEL"]["LAYERS"] = []

    layer = dict()
    if int(extracted_values['gru']) != 0:
        layer["GRU"] = None
        layer["OUT_FEATURES"] = int(extracted_values['horizon']) ** 2
        layer["LAYERS"] = int(extracted_values['gru'])
        param_dict["MODEL"]["LAYERS"].append(layer)
    for i in range(int(extracted_values['layers'])):
        layer = dict()
        layer["DENSE"] = None
        layer["OUT_FEATURES"] = int(extracted_values['neurons'])
        layer["ACTIVATION"] = "Mish"
        param_dict["MODEL"]["LAYERS"].append(layer)
    param_dict["MODEL"]["OPTIMIZATION"]["BATCH_SIZE"] = int(extracted_values['batch'])
    param_dict["MODEL"]["OPTIMIZATION"]["LR"] = float(extracted_values['lr'])
    param_dict["MODEL"]["HORIZON"] = int(extracted_values['horizon'])
    model = string_to_model[param_dict["MODEL"]["NAME"]](param_dict)
    
    if plot_original:
        data_file = os.path.join(dir_name,"data/original_data/Putnam_park2023_run4_2_{}RNN_Val.npz".format(extracted_values['horizon']))
    else:
        
        data_file = os.path.join(dir_name,"data/ekf_model_data/denoised_csv_{}RNN_val.npz".format(extracted_values['horizon']))
    print("data file name:",data_file)
    
    data_npy = np.load(data_file)

    dataset = string_to_dataset[param_dict["MODEL"]["NAME"]](data_npy["features"], data_npy["labels"],
                                                            data_npy["times_features"],data_npy["times"])
    
    val_data_loader = torch.utils.data.DataLoader(dataset, batch_size=int(extracted_values['batch']), shuffle=False, drop_last=True)
    model.cuda()
    model.eval()
    if model.is_rnn:
        val_h = model.init_hidden(model.batch_size)
    
    s_qualify = []
    s_label = []
    time = []
    
    
    for val_f, val_la, val_timef, val_times, val_normin in val_data_loader:

        val_f, val_la, val_times = val_f.to(device), val_la.to(device), val_times.to(device)
        val_timef = val_timef.to(device)
        val_normin = val_normin.to(device)
        if model.is_rnn:
            val_h = val_h.data
            out, val_h, _, force_qualify = model(val_f, val_normin, val_h,val_timef, val_times)
        else:
            out, _, _, force_qualify = model(val_f, val_normin,None, val_timef, val_times)
            

        speed_qualify = out.detach().cpu().numpy()
        l_val = val_la.detach().cpu().numpy()
        time_la = val_timef.detach().cpu().numpy()
        
        s_qualify.append(speed_qualify)
        s_label.append(l_val)
        time.append(time_la)

    speed_qualify = convert_list_to_array_correct(s_qualify)
    time_qualify = convert_list_to_array_correct(time)
    
    print(speed_qualify.shape)
    print("time shape :",time_qualify.shape)
    
    
    vx_quailify = speed_qualify[:,0]
    vy_quailify = speed_qualify[:,1]
    yawRate_quailify = speed_qualify[:,2]

    label = convert_list_to_array_correct(s_label)
    print(label.shape)
    vx_label = label[:,0]
    vy_label = label[:,1]
    yawRate_label = label[:,2]
    if not ground_truth:
        axs[0].plot(time_qualify,vx_label,color='orange',linestyle='-',label='GT', linewidth=plot_linewidth)
        axs[1].plot(time_qualify,vy_label,color='orange',linestyle='-',label='GT', linewidth=plot_linewidth)
        axs[2].plot(time_qualify,vx_label,color='orange',linestyle='-',label='GT', linewidth=plot_linewidth)
        axs[3].plot(time_qualify,vy_label,color='orange',linestyle='-',label='GT', linewidth=plot_linewidth) 
        ground_truth = True
        
    error_max = np.max(np.abs(vy_label - vy_quailify))
    
    error_idx = np.argmax(np.abs(vy_label - vy_quailify))
    
    zoom_time = time_qualify[error_idx]
    
    error_vx_max = np.max(np.abs(vx_label - vx_quailify))
    
    error_vx_idx = np.argmax(np.abs(vx_label - vx_quailify))
    
    zoom_vx_time = time_qualify[error_vx_idx]
    zoom_vx_lower = vx_quailify[error_vx_idx]
    
    zoom_vx_upper = vx_label[error_vx_idx]
    
    print("error vy max: {}, position: {}, time: {}".format(error_max, error_idx, zoom_time))
    print("error vx max: {}, position: {}, time: {}".format(error_vx_max, error_vx_idx, zoom_vx_time))
    
        
    
    
    if plot_original:
        axs[0].plot(time_qualify,vx_quailify,color=color[extracted_values['p']],linestyle=linestyle[extracted_values['p']],label='DDM {}%'.format(extracted_values['p']), linewidth=plot_linewidth)
        axs[1].plot(time_qualify,vy_quailify,color=color[extracted_values['p']],linestyle=linestyle[extracted_values['p']],label='DDM {}%'.format(extracted_values['p']), linewidth=plot_linewidth)
        axs[2].plot(time_qualify,vx_quailify,color=color[extracted_values['p']],linestyle=linestyle[extracted_values['p']],label='DDM {}%'.format(extracted_values['p']), linewidth=plot_linewidth)
        axs[3].plot(time_qualify,vy_quailify,color=color[extracted_values['p']],linestyle=linestyle[extracted_values['p']],label='DDM {}%'.format(extracted_values['p']), linewidth=plot_linewidth)
     
    else:
        # axs[0].plot(time_qualify,vy_quailify,color=color[j],linestyle='-.',label='Hybrid Adam Vy {}%'.format(extracted_values['p']))
        # axs[1].plot(Fry[:,0],Fry[:,1],color=color[j],linestyle='-.',label='Hybrid Adam Fry {}%'.format(extracted_values['p']))
        # axs[2].plot(Ffy[:,0],Ffy[:,1],color=color[j],linestyle='-.',label='Hybrid Adam Ffy {}%'.format(extracted_values['p']))
        # axs[3].plot(Fry[:,0],Fry[:,1],color=color[j],linestyle='-.',label='Hybrid Adam Fry {}%'.format(extracted_values['p']))
        axs[0].plot(time_qualify,vx_quailify,color=color[extracted_values['p']],linestyle=linestyle[extracted_values['p']],label='FTHD {}%'.format(extracted_values['p']), linewidth=plot_linewidth)
        axs[1].plot(time_qualify,vy_quailify,color=color[extracted_values['p']],linestyle=linestyle[extracted_values['p']],label='FTHD {}%'.format(extracted_values['p']), linewidth=plot_linewidth)
        axs[2].plot(time_qualify,vx_quailify,color=color[extracted_values['p']],linestyle=linestyle[extracted_values['p']],label='FTHD {}%'.format(extracted_values['p']), linewidth=plot_linewidth)
        axs[3].plot(time_qualify,vy_quailify,color=color[extracted_values['p']],linestyle=linestyle[extracted_values['p']],label='FTHD {}%'.format(extracted_values['p']), linewidth=plot_linewidth)
    

if plot_original:
    axs[0].set_xlim(15,475) 
else:
    axs[0].set_xlim(15,475)
    
    
# axs[0].set_ylim(0.1875,0.225)
# axs[0].set_title('Vx with {} percent trainset'.format(training_percent),fontsize=fontsize)
axs[0].legend(loc='lower right')

if plot_original:
    axs[1].set_xlim(15,475)
else:
    axs[1].set_xlim(15,475)
    
axs[1].set_ylim(-1.0,2.0)
# axs[1].set_title('Vy force with {} percent trainset'.format(training_percent),fontsize=fontsize)
axs[1].legend(loc='lower right')

axs[2].set_xlim(zoom_vx_time-0.04,zoom_vx_time+0.04)
axs[2].set_ylim(np.minimum(zoom_vx_upper,zoom_vx_lower)-0.1,np.maximum(zoom_vx_lower,zoom_vx_upper)+0.1)
# axs[2].set_title('Zoomed Vx where the maximum error happens',fontsize=fontsize)
axs[2].legend(loc='upper left')

axs[3].set_xlim(zoom_time-5,zoom_time+5)
if plot_original:
    axs[3].set_ylim(-0.71,1.3)
else:
    axs[3].set_ylim(-0.32,0.176)
    
# axs[3].set_title('Zoomed Vy where the maximum error happens',fontsize=fontsize)
axs[3].legend(loc='lower right')

axs[0].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
axs[0].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)

axs[1].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
axs[1].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

axs[2].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
axs[2].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

axs[3].grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
axs[3].grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

axs[0].set_ylabel('$V_x$ (m/s)',fontsize=fontsize)
axs[2].set_ylabel('$V_x$ (m/s)',fontsize=fontsize)

axs[1].set_ylabel('$V_y$ (m/s)',fontsize=fontsize)
axs[3].set_ylabel('$V_y$ (m/s)',fontsize=fontsize)


for i in range(4):
    axs[i].set_xlabel('Time (s)',fontsize=fontsize)
    axs[i].tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
    axs[i].legend(fontsize=legend_fontsize, handlelength=1.0, handletextpad=0.2, borderaxespad=0.2)
    
plt.tight_layout()

if plot_original:
    saved_file = 'output/model_speed_benchmark_DDM.svg'
    pdf_save_file = 'output/model_speed_benchmark_DDM.pdf'
else:
    saved_file = 'output/model_speed_benchmarkEKF_FTHD.svg'
    pdf_save_file = 'output/model_speed_benchmarkEKF_FTHD.pdf'
    
plt.savefig(os.path.join(dir_name,saved_file),transparent=True, format='svg')
plt.savefig(os.path.join(dir_name,pdf_save_file),transparent=True, format='pdf')


plt.show()