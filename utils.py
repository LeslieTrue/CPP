def update_pi_from_z(net):
    import copy
    model_dict = net.state_dict()
    save_dict = copy.deepcopy(model_dict)
    to_rename_keys = []
    for key in save_dict:
        if 'subspace' in key:
            to_rename_keys.append(key)
    for key in to_rename_keys:
        print(f'renamed key {key}')
        pre, post = key.split('subspace')
        save_dict[pre + 'cluster' + post] = save_dict.pop(key)

    model_dict.update(save_dict)
    log = net.load_state_dict(model_dict)
    print(log)
    return net