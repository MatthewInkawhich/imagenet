
    # Next, get all unique valid prefixes
    prefixes = []
    for i in range(len(valid_combos)):
        for j in range(len(valid_combos[i])):
            prefixes.append(valid_combos[i][:j])
    prefixes = list(set(prefixes))
    prefixes.sort()  # Sort by value
    prefixes.sort(key=lambda t: len(t)) # Sort by length
    #print("\nvalid prefixes:", prefixes)

    # Initialize and fill the selector_truth lookup
    # Every sample has an entry that is a dict of (prefix:true next stride option)
    print("Creating selector_truth lookup...")
    selector_truth = [{} for i in range(dataset_length)]
    for sample_idx in range(dataset_length):
        # Start new dict for this sample
        curr_dict = {}
        # Iterate over prefixes
        for prefix in prefixes:
            #print("\nprefix:", prefix)
            # Get coord of min loss for the current prefix
            prefix_with_sample_idx = (sample_idx,) + prefix
            # Record the truth_idx (we want to find the stride option that comes immediately AFTER prefix)
            truth_idx = len(prefix_with_sample_idx)
            #print("prefix_with_sample_idx:", prefix_with_sample_idx)
            min_idx = torch.argmin(all_losses[prefix_with_sample_idx])
            #print("size_of_argmin_tensor:", tuple(all_losses[prefix_with_sample_idx].shape))

            #print("min_idx:", min_idx)
            min_coord = unravel_index(min_idx.item(), tuple(all_losses[prefix_with_sample_idx].shape))
            min_coord_full = prefix_with_sample_idx + min_coord
            #print("min_coord:", min_coord)
            #print("min_coord_full:", min_coord_full)

            # The truth is the index value corresponding to the truth_idx
            #print("truth_idx:", truth_idx)
            truth = min_coord_full[truth_idx]
            #print("truth:", truth)
            # Update current sample's dict
            curr_dict[prefix] = truth

        selector_truth[sample_idx].update(curr_dict)


    #print("\n\nselector_truth:")
    #for d_idx in range(len(selector_truth)):
    #    print(d_idx)
    #    for k, v in selector_truth[d_idx].items():
    #        print(k, v)
    #exit()
