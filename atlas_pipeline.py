"""A medical image analysis pipeline.

The pipeline is used for brain tissue segmentation using atlas based segmentation
"""
import argparse
import datetime
import os
import sys
import timeit
import warnings

import SimpleITK as sitk
import sklearn.ensemble as sk_ensemble
import numpy as np
import pymia.data.conversion as conversion
import pymia.evaluation.writer as writer

try:
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil
except ImportError:
    # Append the MIALab root directory to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[0]), '..'))
    import mialab.data.structure as structure
    import mialab.utilities.file_access_utilities as futil
    import mialab.utilities.pipeline_utilities as putil

LOADING_KEYS = [structure.BrainImageTypes.T1w,
                structure.BrainImageTypes.T2w,
                structure.BrainImageTypes.GroundTruth,
                structure.BrainImageTypes.BrainMask,
                structure.BrainImageTypes.RegistrationTransform]  # the list of data we will load


def main(result_dir: str, data_atlas_dir: str, data_train_dir: str, data_test_dir: str):
    """Brain tissue segmentation using atlas based segmentation.

    The main routine executes the medical image analysis pipeline:

        - Image loading
        - Registration
        - Pre-processing
        - Atlas Lable prediction via majority voting
        - Segmentation using the atlaslabels after registering to unseen images
        - Post-processing of the segmentation (?)
        - Evaluation of the segmentation
    """

    # load atlas images
    putil.load_atlas_images(data_atlas_dir)

    print('-' * 5, 'Training...')

    # crawl the training image directories
    crawler = futil.FileSystemDataCrawler(data_train_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())
    pre_process_params = {'skullstrip_pre': True,
                          'normalization_pre': True,
                          'registration_pre': True,
                          'coordinates_feature': True,
                          'intensity_feature': True,
                          'gradient_intensity_feature': True}

    # load images for training and pre-process
    images = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    # generate feature matrix and label vector
    data_train = np.concatenate([img.feature_matrix[0] for img in images])
    labels_train = np.concatenate([img.feature_matrix[1] for img in images]).squeeze()
    
    ###majority voting BEGIN
    # generate atlas labels with training image labels via majority voting
    # --- Generate atlas labels by majority voting ---
    warnings.warn('-- generating atlas labels via majority voting')

    warped_labels = []

    for img in images:
        # ground truth label
        gt = img.images[structure.BrainImageTypes.GroundTruth]

        # registration transform TRAIN → ATLAS computed in preprocessing (check that exists)
        if structure.BrainImageTypes.RegistrationTransform not in img.images:
            raise ValueError(f"Missing RegistrationTransform for subject {img.id_}")
        tx = img.images[structure.BrainImageTypes.RegistrationTransform]

        # warp GT from training image → atlas space
        gt_atlas = sitk.Resample(
            gt,
            putil.atlas_t1,                 # atlas reference space
            tx,                             # image → atlas transform
            sitk.sitkNearestNeighbor,       # preserves discrete labels
            0,
            sitk.sitkUInt8
        )

        warped_labels.append(sitk.GetArrayFromImage(gt_atlas))

    # majority vote per voxel ----------------------------------------------------------------to experiment
    label_stack = np.stack(warped_labels, axis=0)
    atlas_labels = np.apply_along_axis(
        lambda x: np.bincount(x.astype(np.uint8)).argmax(),
        axis=0,
        arr=label_stack
    ).astype(np.uint8)

    atlas_labels_img = sitk.GetImageFromArray(atlas_labels)
    atlas_labels_img.CopyInformation(putil.atlas_t1)
    ### majority voting END 
    '''
    #### STAPLE label generation BEGIN ####
    warnings.warn('-- generating atlas labels via Staple fusion')

    # Use STAPLE (SimpleITK) to fuse binary masks per label into probabilistic maps
    
    #### STAPLE label generation BEGIN ####
    warnings.warn('-- generating atlas labels via Staple fusion')

    # warp all GTs into atlas space first
    warped_gt_images = []
    for img in images:
        gt = img.images[structure.BrainImageTypes.GroundTruth]
        if structure.BrainImageTypes.RegistrationTransform not in img.images:
            raise ValueError(f"Missing RegistrationTransform for subject {img.id_}")
        tx = img.images[structure.BrainImageTypes.RegistrationTransform]

        gt_atlas = sitk.Resample(
            gt,
            putil.atlas_t1,
            tx,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8
        )
        warped_gt_images.append(gt_atlas)

    # collect label values present in the warped GTs   #### TODO rename w registered_gt? and lv -> label
    label_values = set()
    for w in warped_gt_images:
        vals = np.unique(sitk.GetArrayFromImage(w))
        for v in vals:
            label_values.add(int(v))
    label_values = sorted(label_values)    ### try label_values = set(0,1,2,3,4,5)

    if len(label_values) == 0:
        raise RuntimeError("No label values found in warped ground truths; cannot build atlas labels.")

    # compute a STAPLE probability map for each label value
    prob_maps = []
    for lv in label_values:
        masks = [sitk.Cast(sitk.Equal(w, lv), sitk.sitkUInt8) for w in warped_gt_images]
        if len(masks) == 1:
            prob = sitk.Cast(masks[0], sitk.sitkFloat32)
        else:
            staple = sitk.STAPLEImageFilter()
            prob = staple.Execute(masks)  # float image in [0,1]
        prob_maps.append(sitk.GetArrayFromImage(prob))

    if not prob_maps:
        raise RuntimeError("STAPLE produced no probability maps; aborting atlas label generation.")

    # stack probability maps and select label with max probability per voxel
    prob_stack = np.stack(prob_maps, axis=0)  # shape (n_labels, z, y, x)
    argmax_idx = np.argmax(prob_stack, axis=0)  # shape (z, y, x)
    label_values_arr = np.array(label_values, dtype=np.uint8)
    atlas_labels = label_values_arr[argmax_idx]

    atlas_labels_img = sitk.GetImageFromArray(atlas_labels.astype(np.uint8))
    atlas_labels_img.CopyInformation(putil.atlas_t1)
    #### STAPLE label generation END ####
    '''

    # store in pipeline utilities so later steps can access it
    putil.atlas_labels = atlas_labels_img

    # create a result directory with timestamp
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    result_dir = os.path.join(result_dir, t)
    os.makedirs(result_dir, exist_ok=True)

    print('-' * 5, 'Testing...')

    # initialize evaluator
    evaluator = putil.init_evaluator()

    # crawl the test image directories
    crawler = futil.FileSystemDataCrawler(data_test_dir,
                                          LOADING_KEYS,
                                          futil.BrainImageFilePathGenerator(),
                                          futil.DataDirectoryFilter())

    # load images for testing and pre-process
    pre_process_params['training'] = False
    images_test = putil.pre_process_batch(crawler.data, pre_process_params, multi_process=False)

    images_prediction = []  
    images_probabilities = []  # dummy probabilities for post-processing    

    for img in images_test:
        print('-' * 10, 'Testing', img.id_)

        start_time = timeit.default_timer()
        ##segment test images with atlas labels (-> prediction)
        # --- Segment test image using atlas labels ---
        # 1. Get the TEST → ATLAS transform from preprocessing
        tx_test_to_atlas = img.images[structure.BrainImageTypes.RegistrationTransform]

        # 2. Invert it: we need ATLAS → TEST
        tx_atlas_to_test = tx_test_to_atlas.GetInverse()

        # 3. Warp atlas labels into the test image space
        predictions_img = sitk.Resample(
            putil.atlas_labels,
            img.images[structure.BrainImageTypes.T1w],   # test image reference
            tx_atlas_to_test,
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8
        )

        # convert to numpy for further processing
        predictions = sitk.GetArrayFromImage(predictions_img)
        probabilities = np.ones_like(predictions, dtype=np.float32)  # dummy probabilities (not calculated in atlas-based segmentation)
        print(' Time elapsed:', timeit.default_timer() - start_time, 's')

        # convert prediction and probabilities back to SimpleITK images
        image_prediction = conversion.NumpySimpleITKImageBridge.convert(predictions.astype(np.uint8),
                                                                        img.image_properties)
        image_probabilities = conversion.NumpySimpleITKImageBridge.convert(probabilities, img.image_properties)

        # evaluate segmentation without post-processing
        evaluator.evaluate(image_prediction, img.images[structure.BrainImageTypes.GroundTruth], img.id_)

        images_prediction.append(image_prediction)
        images_probabilities.append(image_probabilities)  ### for post-processing (all one maps)

                # --- SAVE PREDICTED LABEL MASKS ---
        # Make directory "predictions" inside result_dir
        pred_dir = os.path.join(result_dir, "predicted_labels")
        os.makedirs(pred_dir, exist_ok=True)

        # File name: <subject_id>_SEG.mha
        save_path = os.path.join(pred_dir, f"{img.id_}_SEG.mha")

        # Save the SimpleITK prediction image
        sitk.WriteImage(image_prediction, save_path, True)

        print(f"Saved prediction: {save_path}")

    ### POSTPROCESSING ###
    #post-process segmentation and evaluate with post-processing
    post_process_params = {'simple_post': True} #means we access the post-processing class in pipeline utilities
    images_post_processed = putil.post_process_batch(images_test, images_prediction, images_probabilities, post_process_params, multi_process=False)

    for i, img in enumerate(images_test):
        evaluator.evaluate(images_post_processed[i], img.images[structure.BrainImageTypes.GroundTruth],img.id_ + '-PP')

        # save results
        sitk.WriteImage(images_prediction[i], os.path.join(result_dir, images_test[i].id_ + '_SEG.mha'), True)
        sitk.WriteImage(images_post_processed[i], os.path.join(result_dir, images_test[i].id_ + '_SEG-PP.mha'), True)
    ### POSTPROCESSING ###

    # use two writers to report the results
    os.makedirs(result_dir, exist_ok=True)  # generate result directory, if it does not exists
    result_file = os.path.join(result_dir, 'results.csv')
    writer.CSVWriter(result_file).write(evaluator.results)

    print('\nSubject-wise results...')
    writer.ConsoleWriter().write(evaluator.results)

    # report also mean and standard deviation among all subjects
    result_summary_file = os.path.join(result_dir, 'results_summary.csv')
    functions = {'MEAN': np.mean, 'STD': np.std}
    writer.CSVStatisticsWriter(result_summary_file, functions=functions).write(evaluator.results)
    print('\nAggregated statistic results...')
    writer.ConsoleStatisticsWriter(functions=functions).write(evaluator.results)

    # clear results such that the evaluator is ready for the next evaluation
    evaluator.clear()


if __name__ == "__main__":
    """The program's entry point."""

    script_dir = os.path.dirname(sys.argv[0])

    parser = argparse.ArgumentParser(description='Medical image analysis pipeline for brain tissue segmentation')

    parser.add_argument(
        '--result_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './mia-result')),
        help='Directory for results.'
    )

    parser.add_argument(
        '--data_atlas_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './data/atlas')),
        help='Directory with atlas data.'
    )

    parser.add_argument(
        '--data_train_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './data/train/')),
        help='Directory with training data.'
    )

    parser.add_argument(
        '--data_test_dir',
        type=str,
        default=os.path.normpath(os.path.join(script_dir, './data/test/')),
        help='Directory with testing data.'
    )

    args = parser.parse_args()
    main(args.result_dir, args.data_atlas_dir, args.data_train_dir, args.data_test_dir)
