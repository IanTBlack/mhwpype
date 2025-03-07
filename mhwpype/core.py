import pandas as pd
import xarray as xr

def annual_circular_smoother(da: xr.DataArray,
                             half_window_width: int) -> xr.DataArray:

    """
    This function allows for the rolling smoothing of the beginning and end of a climatology or threshold dataset,
    which might otherwise be unsmoothed.

    Pitfalls: This function only works if the input climatology/threshold data has 366 days.
    If only a part of a climatology dataset is provided (e.g. dayofyear 250-350),
    then this function will inappropriately smooth the original dataset with the prepended and appended data.
    If a subset of the climatology is needed, it is recommended that you subset after circular smoothing.

    :param da: A dataset that represents a climatology or threshold dataset with a dayofyear coordinate.
    :param half_window_width: The number of days to include on either side of the central day for smoothing.
    :return: A smoothed dataset with the same dayofyear values as the original dataset.
    """

    # Create a new dataset which can be prepended to the original for "circular" smoothing.
    pre = da.copy(deep=True)
    pre['dayofyear'] = da.dayofyear.min() - da.dayofyear
    pre = pre.sortby('dayofyear')

    # Create a new dataset which can be appended to the original data for "circular" smoothing.
    post = da.copy(deep=True)
    post['dayofyear'] = da.dayofyear.max() + da.dayofyear
    post = post.sortby('dayofyear')

    # Combine the original data with the prepended and appended data.
    circ = xr.combine_by_coords([pre, da, post])

    # Smooth the data with a rolling mean using the half window width.
    rcda = circ.rolling({'dayofyear': 2 * half_window_width + 1}, center=True).mean(skipna=True)
    rcda = rcda.sel(dayofyear=da.dayofyear)  # Select the original dayofyear values for return.
    return rcda


def group_mhw_events(mask: xr.DataArray) -> xr.Dataset:
    all_events = []
    for _lat in mask['latitude']:
        for _lon in mask['longitude']:
            for _depth in mask['depth']:
                for _q in mask['quantile']:
                    cell_data = mask.sel(latitude = _lat, longitude = _lon, depth = _depth, quantile = _q)
                    cell_mask = cell_data.where(cell_data == 1, drop = True)
                    cell_dts = sorted(cell_mask.time.values)
                    grp = [cell_dts[0]]
                    cell_groups = []
                    for i in range(len(cell_dts)):
                        if cell_dts[i] == grp[-1] + np.timedelta64(86400,'s'):
                            grp.append(cell_dts[i])
                        else:
                            grp = [cell_dts[i]]
                        ts = pd.to_datetime(min(grp)).replace(hour = 0, minute = 0, second = 0)
                        te = pd.to_datetime(max(grp)).replace(hour = 23, minute = 59, second = 59)
                        gds = xr.Dataset()
                        gds = gds.assign_coords({'event': [ts]})
                        gds['event_start'] = (['event'],[ts])
                        gds['event_end'] = (['event'],[te])
                        cell_groups.append(gds)
                    cell_events = xr.combine_by_coords(cell_groups)
                    cell_events = cell_events.expand_dims({'quantile': [_q], 'latitude': [_lat], 'longitude': [_lon], 'depth': [_depth]})
                    all_events.append(cell_events)
    grouped_events = xr.combine_by_coords(all_events)
    return grouped_events