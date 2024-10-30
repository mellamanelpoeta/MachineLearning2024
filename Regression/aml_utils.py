import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

visualize = True
perform_computation = True
test_db_dir = os.path.dirname(os.path.realpath(__file__)) + f'/test_db'

def retrieve_item(item_name, ptr_, test_idx, npz_file):
    item_shape = npz_file[f'shape_{item_name}'][test_idx]
    item_size = int(np.prod(item_shape))
    item = npz_file[item_name][ptr_:(ptr_+item_size)].reshape(item_shape)
    return item, ptr_+item_size

class NPStrListCoder:
    def __init__(self):
        self.filler = '?'
        self.spacer = ':'
        self.max_len = 100
    
    def encode(self, str_list):
        my_str_ = self.spacer.join(str_list)
        str_hex_data = [ord(c) for c in my_str_]
        assert_msg = f'Increase max len; you have so many characters: {len(str_hex_data)}>{self.max_len}'
        assert len(str_hex_data) <= self.max_len, assert_msg
        str_hex_data = str_hex_data + [ord(self.filler) for _ in range(self.max_len - len(str_hex_data))]
        str_hex_np = np.array(str_hex_data)
        return str_hex_np
    
    def decode(self, np_arr):
        a = ''.join([chr(i) for i in np_arr])
        recovered_list = a.replace(self.filler, '').split(self.spacer)
        return recovered_list
    
str2np_coder = NPStrListCoder()

def test_case_loader(test_file):
    
    npz_file = np.load(test_file)
    arg_id_list = sorted([int(key[4:]) for key in npz_file.keys() if key.startswith('arg_')])
    kwarg_names_list = sorted([key[6:] for key in npz_file.keys() if key.startswith('kwarg_')])

    arg_ptr_list = [0 for _ in range(len(arg_id_list))]
    dfcarg_ptr_list = [0 for _ in range(len(arg_id_list))]
    kwarg_ptr_list = [0 for _ in range(len(kwarg_names_list))]
    dfckwarg_ptr_list = [0 for _ in range(len(kwarg_names_list))]
    out_ptr = 0
    for i in np.arange(npz_file['num_tests']):
        args_list = []
        for arg_id, arg_id_ in enumerate(arg_id_list):
            arg_item, arg_ptr_list[arg_id] = retrieve_item(f'arg_{arg_id_}', arg_ptr_list[arg_id], i, npz_file)
            if f'dfcarg_{arg_id_}' in npz_file.keys():
                col_list_code, dfcarg_ptr_list[arg_id] = retrieve_item(f'dfcarg_{arg_id_}', dfcarg_ptr_list[arg_id], i, npz_file)
                arg_item = pd.DataFrame(arg_item, columns=str2np_coder.decode(col_list_code))
            args_list.append(arg_item)
        args = tuple(args_list)

        kwargs = {}
        for kwarg_id, kwarg_name in enumerate(kwarg_names_list):
            kwarg_item, kwarg_ptr_list[kwarg_id] = retrieve_item(f'kwarg_{kwarg_name}', kwarg_ptr_list[kwarg_id], i, npz_file)
            if f'dfckwarg_{kwarg_name}' in npz_file.keys():
                col_list_code, dfckwarg_ptr_list[kwarg_id] = retrieve_item(f'dfckwarg_{kwarg_name}', dfckwarg_ptr_list[kwarg_id], i, npz_file)
                kwarg_item = pd.DataFrame(kwarg_item, columns=str2np_coder.decode(col_list_code))
            kwargs[kwarg_name]=kwarg_item

        output, out_ptr = retrieve_item(f'output', out_ptr, i, npz_file)

        yield args, kwargs, output
        
def arg2str(args, kwargs, adv_user_msg=False):
    msg  = f'Following are the test case arguments that were used to help you diagnose the issue.\n'
    msg += f'Each argument will be printed on a separate line: \n\n\n'
    if adv_user_msg:
        msg += f'Note:\n'
        msg += f'  * If the data input is too large, it might get truncated and you might not see the whole arguments in the printed message.\n'
        msg += f'  * If the input had high-precision numbers, the printing precision may not be enough to reproduce the same exact output.\n'
        msg += f' In these cases, you should follow the instructions below (i.e., the alternative approach section).\n\n\n\n'
        
    for arg_ in args:
        msg += f'{arg_},\n'
    for key,val in kwargs.items():
        try:
            val_str = np.array_repr(val)
        except:
            val_str = val
        new_line = f'{key}={val_str},\n'
        new_line = new_line.replace('=array(','=np.array(')
        new_line = new_line.replace('nan,','np.nan,')
        msg += new_line
    if adv_user_msg:
        msg += '\n\n---------\n'
        msg += 'Alternative approach (recommended):\n  In case you would rather not copy the arguments from above or the printing percision was causing distortions/truncations, the test results dictionary, which was returned, contains the following material :\n\n'
        msg += "    1) Arguments tuple passed to your function ==> test_results['test_args']\n"
        msg += "    2) Keyword arguments dictionary passed to your function ==> test_results['test_kwargs']\n"
        msg += "    3) The correct solution ==> test_results['correct_sol']\n"
        msg += "    4) Your function's solution ==> test_results['stu_sol']\n\n"
        msg += "  Therefore, you should expect the following tests to pass if your implementation was correct:\n\n"
        msg += "    assert np.array_equal(test_results['correct_sol'], test_results['stu_sol'])\n"
        msg += "    assert np.array_equal(test_results['stu_sol'], YOUR_FUNCTION_NAME(*test_results['test_args'], **test_results['test_kwargs']))\n"
    return msg


def test_case_checker(stu_func, task_id=0):
    out_dict = {}
    out_dict['task_number'] = task_id
    test_db_npz = f'{test_db_dir}/task_{task_id}.npz'
    if not os.path.exists(test_db_npz):
        out_dict['message'] = f'Test database test_db/task_{task_id}.npz does not exist... aborting!'
        out_dict['passed'] = False
        out_dict['test_args'] = None
        out_dict['test_kwargs'] = None
        out_dict['stu_sol'] = None
        out_dict['correct_sol'] = None
        return None
    done = False
    for (test_args, test_kwargs, correct_sol) in test_case_loader(test_db_npz):
        try:
            stu_sol = stu_func(*test_args, **test_kwargs)
        except:
            stu_sol = None
            message =  f'\nError in task {task_id}: Your code raised an exception (a.k.a. a fatal error in python). \n' 
            message += f'We will give you the test case arugments which caused such an error. '
            message += f'You can run your code on this test case on your own, find the exact error, '
            message += f'diagnose the issue, and fix it. \n'
            message += f'----------\n'
            message += arg2str(test_args, test_kwargs, adv_user_msg=True)
            out_dict['test_args'] = test_args
            out_dict['test_kwargs'] = test_kwargs
            out_dict['stu_sol'] = stu_sol
            out_dict['correct_sol'] = correct_sol
            out_dict['message'] = message
            out_dict['passed'] = False
            return out_dict
        
        if isinstance(correct_sol, np.ndarray) and np.isscalar(stu_sol):
            # This is handling a special case: When scalar numpy objects are stored, 
            # they will be converted to a numpy array upon loading. 
            # In this case, we'll give students the benefit of the doubt, 
            # and assume the correct solution already was a scalar.
            if correct_sol.size == 1:
                correct_sol = np.float64(correct_sol.item())
                stu_sol = np.float64(np.float64(stu_sol).item())
        
        #Type Sanity check
        if type(stu_sol) is not type(correct_sol):
            message =  f'\nError in task {task_id}: Your solution\'s output type is not the same as '
            message += f'the reference solution\'s data type.\n' 
            message += f'    Your solution\'s type --> {type(stu_sol)}\n'
            message += f'    Correct solution\'s type --> {type(correct_sol)}\n'
            message += f'----------\n'
            message += arg2str(test_args, test_kwargs, adv_user_msg=True)
            out_dict['test_args'] = test_args
            out_dict['test_kwargs'] = test_kwargs
            out_dict['stu_sol'] = stu_sol
            out_dict['correct_sol'] = correct_sol
            out_dict['message'] = message
            out_dict['passed'] = False
            return out_dict
        
        if isinstance(correct_sol, np.ndarray):
            if not np.all(np.array(correct_sol.shape) == np.array(stu_sol.shape)):
                message  = f'\nError in task {task_id}: Your solution\'s output numpy shape is not the same as '
                message += f'the reference solution\'s shape.\n'
                message += f'    Your solution\'s shape --> {stu_sol.shape}\n'
                message += f'    Correct solution\'s shape --> {correct_sol.shape}\n'
                message += f'----------\n'
                message += arg2str(test_args, test_kwargs, adv_user_msg=True)
                out_dict['test_args'] = test_args
                out_dict['test_kwargs'] = test_kwargs
                out_dict['stu_sol'] = stu_sol
                out_dict['correct_sol'] = correct_sol
                out_dict['message'] = message
                out_dict['passed'] = False
                return out_dict
            
            if not(stu_sol.dtype is correct_sol.dtype):
                message  = f'\nError in task {task_id}: Your solution\'s output numpy dtype is not the same as'
                message += f'the reference solution\'s dtype.\n'
                message += f'    Your solution\'s dtype --> {stu_sol.dtype}\n'
                message += f'    Correct solution\'s dtype --> {correct_sol.dtype}\n'
                message += f'----------\n'
                message += arg2str(test_args, test_kwargs, adv_user_msg=True)
                out_dict['test_args'] = test_args
                out_dict['test_kwargs'] = test_kwargs
                out_dict['stu_sol'] = stu_sol
                out_dict['correct_sol'] = correct_sol
                out_dict['message'] = message
                out_dict['passed'] = False
                return out_dict
        
        if isinstance(correct_sol, np.ndarray):
            equality_array = np.isclose(stu_sol, correct_sol, rtol=1e-05, atol=1e-08, equal_nan=True)
            if not equality_array.all():
                message = f'\nError in task {task_id}: Your solution is not the same as the correct solution. '
                message += 'The following is the issue...\n'
                whr_ = np.array(np.where(np.logical_not(equality_array)))
                ineq_idxs = whr_[:,0].tolist()
                message += f'    your_solution{ineq_idxs}={stu_sol[tuple(ineq_idxs)]}\n'
                message += f'    correct_solution{ineq_idxs}={correct_sol[tuple(ineq_idxs)]}\n'
                message += f'----------\n'
                message += arg2str(test_args, test_kwargs, adv_user_msg=True)
                out_dict['test_args'] = test_args
                out_dict['test_kwargs'] = test_kwargs
                out_dict['stu_sol'] = stu_sol
                out_dict['correct_sol'] = correct_sol
                out_dict['message'] = message
                out_dict['passed'] = False
                return out_dict
            
        elif np.isscalar(correct_sol):
            equality_array = np.isclose(stu_sol, correct_sol, rtol=1e-05, atol=1e-08, equal_nan=True)
            if not equality_array.all():
                message = f'\nError in task {task_id}: Your solution is not the same as the correct solution.\n'
                message += f'    your_solution={stu_sol}\n'
                message += f'    correct_solution={correct_sol}\n'
                message += f'----------\n'
                message += arg2str(test_args, test_kwargs, adv_user_msg=True)
                out_dict['test_args'] = test_args
                out_dict['test_kwargs'] = test_kwargs
                out_dict['stu_sol'] = stu_sol
                out_dict['correct_sol'] = correct_sol
                out_dict['message'] = message
                out_dict['passed'] = False
                return out_dict
        
        elif isinstance(correct_sol, tuple):
            if not correct_sol==stu_sol:
                message = f'\nError in task {task_id}: Your solution is not the same as the correct solution.\n'
                message += f'    your_solution={stu_sol}\n'
                message += f'    correct_solution={correct_sol}\n'
                message += f'----------\n'
                message += arg2str(test_args, test_kwargs, adv_user_msg=True)
                out_dict['test_args'] = test_args
                out_dict['test_kwargs'] = test_kwargs
                out_dict['stu_sol'] = stu_sol
                out_dict['correct_sol'] = correct_sol
                out_dict['message'] = message
                out_dict['passed'] = False
                return out_dict
        
        else:
            raise Exception(f'Not implemented comparison for other data types. sorry!')
        
    out_dict['test_args'] = None
    out_dict['test_kwargs'] = None
    out_dict['stu_sol'] = None
    out_dict['correct_sol'] = None
    out_dict['message'] = 'Well Done!'
    out_dict['passed'] = True
    return out_dict

def show_test_cases(test_func, task_id=0):
    from IPython.display import clear_output
    file_path = f'{test_db_dir}/task_{task_id}.npz'
    npz_file = np.load(file_path)
    orig_images = npz_file['raw_images']
    ref_images = npz_file['ref_images']
    test_images = test_func(orig_images)
    
    visualize_ = visualize and perform_computation
    
    if not np.all(np.array(test_images.shape) == np.array(ref_images.shape)):
        print(f'Error: It seems the test images and the ref images have different shapes. Modify your function so that they both have the same shape.')
        print(f' test_images shape: {test_images.shape}')
        print(f' ref_images shape: {ref_images.shape}')
        return None, None, None, False
    
    if not np.all(np.array(test_images.dtype) == np.array(ref_images.dtype)):
        print(f'Error: It seems the test images and the ref images have different dtype. Modify your function so that they both have the same dtype.')
        print(f' test_images dtype: {test_images.dtype}')
        print(f' ref_images dtype: {ref_images.dtype}')
        return None, None, None, False
    
    for i in range(ref_images.shape[0]):
        if visualize_:
            nrows, ncols = 1, 3
            ax_w, ax_h = 5, 5
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*ax_w, nrows*ax_h))
            axes = np.array(axes).reshape(nrows, ncols)
        
        orig_image = orig_images[i]
        ref_image = ref_images[i]
        test_image = test_images[i]
        
        if visualize_:
            ax = axes[0,0]
            ax.pcolormesh(orig_image, edgecolors='k', linewidth=0.01, cmap='Greys')
            ax.xaxis.tick_top()
            ax.invert_yaxis()
            
            x_ticks = ax.get_xticks(minor=False).astype(np.int)
            x_ticks = x_ticks[x_ticks < orig_image.shape[1]]
            ax.set_xticks(x_ticks + 0.5)
            ax.set_xticklabels((x_ticks).astype(np.int))
            
            y_ticks = ax.get_yticks(minor=False).astype(np.int)
            y_ticks = y_ticks[y_ticks < orig_image.shape[0]]
            ax.set_yticks(y_ticks + 0.5)
            ax.set_yticklabels((y_ticks).astype(np.int))
            
            ax.set_aspect('equal')
            ax.set_title('Raw Image')

            ax = axes[0,1]
            ax.pcolormesh(ref_image, edgecolors='k', linewidth=0.01, cmap='Greys')
            ax.xaxis.tick_top()
            ax.invert_yaxis()
            
            x_ticks = ax.get_xticks(minor=False).astype(np.int)
            x_ticks = x_ticks[x_ticks < ref_image.shape[1]]
            ax.set_xticks(x_ticks+0.5)
            ax.set_xticklabels((x_ticks).astype(np.int))
            
            y_ticks = ax.get_yticks(minor=False).astype(np.int)
            y_ticks = y_ticks[y_ticks < ref_image.shape[0]]
            ax.set_yticks(y_ticks+0.5)
            ax.set_yticklabels((y_ticks).astype(np.int))
            
            ax.set_aspect('equal')
            ax.set_title('Reference Solution Image')

            ax = axes[0,2]
            ax.pcolormesh(test_image, edgecolors='k', linewidth=0.01, cmap='Greys')
            ax.xaxis.tick_top()
            ax.invert_yaxis()
            
            x_ticks = ax.get_xticks(minor=False).astype(np.int)
            x_ticks = x_ticks[x_ticks < test_image.shape[1]]
            ax.set_xticks(x_ticks + 0.5)
            ax.set_xticklabels((x_ticks).astype(np.int))
            
            y_ticks = ax.get_yticks(minor=False).astype(np.int)
            y_ticks = y_ticks[y_ticks < test_image.shape[0]]
            ax.set_yticks(y_ticks + 0.5)
            ax.set_yticklabels((y_ticks).astype(np.int))
            
            ax.set_aspect('equal')
            ax.set_title('Your Solution Image')
        
        if (ref_image==test_image).all():
            if visualize_:
                print('The reference and solution images are the same to a T! Well done on this test case.')
        else:
            print('The reference and solution images are not the same...')
            ineq_idxs = np.array(np.where(ref_image!=test_image))[:,0].tolist()
            print(f'ref_image{ineq_idxs}={ref_image[tuple(ineq_idxs)]}')
            print(f'test_image{ineq_idxs}={test_image[tuple(ineq_idxs)]}')
            if visualize_:
                print('I will return the images so that you will be able to diagnose the issue and resolve it...')
            return (orig_image, ref_image, test_image, False)
            
        if visualize_:
            plt.show()
            input_prompt = '    Enter nothing to go to the next image\nor\n    Enter "s" when you are done to recieve the three images. \n'
            input_prompt += '        **Don\'t forget to do this before continuing to the next step.**\n'
            
            try:
                cmd = input(input_prompt)
            except KeyboardInterrupt:
                cmd = 's'
            
            if cmd.lower().startswith('s'):
                return (orig_image, ref_image, test_image, True)
            else:
                clear_output(wait=True)
        
    return (orig_image, ref_image, test_image, True)