from itertools import product
import numpy as np
import warnings
import pandas as pd
import xarray as xr

from .core import annual_circular_smoother


class HOBDAY16:
    HALF_WINDOW_WIDTH = 5
    MINIMUM_EVENT_LENGTH = 5
    MAXIMUM_GAP_LENGTH = 2
    THRESHOLD = 0.90


class Hobday16():
    def build_daily_climatology(self, temperature: xr.DataArray,
                                half_window_width: int = HOBDAY16.HALF_WINDOW_WIDTH) -> xr.Dataset:
        cds = temperature.groupby('time.dayofyear', restore_coord_dims=True).mean(skipna=True)
        rcds = annual_circular_smoother(cds, half_window_width)
        return rcds

    def _build_daily_threshold(self, temperature: xr.DataArray,
                               threshold_value,
                               half_window_width,
                               threshold_method):
        pds = temperature.groupby('time.dayofyear', restore_coord_dims=True).quantile(threshold_value,
                                                                                      method=threshold_method,
                                                                                      skipna=True)
        rpds = annual_circular_smoother(pds, half_window_width)
        return rpds


    def build_daily_threshold(self,
                              temperature: xr.Dataset,
                              threshold_value: float = HOBDAY16.THRESHOLD,
                              half_window_width: int = HOBDAY16.HALF_WINDOW_WIDTH,
                              threshold_method: str = 'linear',
                              suppress_runtime_warnings: bool = True):
        if suppress_runtime_warnings is True:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rpds = self._build_daily_threshold(temperature, threshold_value, half_window_width, threshold_method)
        else:
            rpds = self._build_daily_threshold(temperature, threshold_value, half_window_width, threshold_method)
        return rpds


    def flag_gt_days(self,da, pda):
        gt_mask = xr.where(da > pda.sel(dayofyear=da.time.dt.dayofyear), 1, 0)
        return gt_mask

    def fill_gaps(self, m, maximum_gap_length):
        _m = m.copy()  # xr.apply_ufunc with vectorize=True does not work with inplace operations. This copy is necessary.
        for i in range(len(m)):
            for gap in range(maximum_gap_length, 0, -1):
                gap_condition = np.array([1] + [0] * gap + [1])
                subset = _m[i:i + len(gap_condition)]
                if np.array_equal(subset, gap_condition):
                    _m[i:i + len(gap_condition)] = 1
        return _m


    def build_mhw_mask(self,da, pda, minimum_event_length: int = 5, maximum_gap_length: int = 2):
        gt_mask = self.flag_gt_days(da,pda)
        gt_windows = gt_mask.rolling({'time': minimum_event_length}).construct('gt_window')
        raw_mhw_mask = xr.where(gt_windows.sum(dim='gt_window') == minimum_event_length, 1, 0)
        shifted = [raw_mhw_mask] + [raw_mhw_mask.shift(time = -i) for i in range(1, minimum_event_length)]
        patched_mhw_mask = xr.where(sum(shifted) >=1, 1, 0)
        updated_mhw_mask = xr.apply_ufunc(self.fill_gaps, patched_mhw_mask,
                                          kwargs = {"maximum_gap_length": maximum_gap_length},
                                          input_core_dims = [['time']],
                                          output_core_dims = [['time']],
                                          vectorize = True)

        #TODO: Change name and attributes of updated_mhw_mask.
        updated_mhw_mask = updated_mhw_mask.drop_vars('dayofyear', errors='ignore')
        return updated_mhw_mask

    def group_cell_events(self, cell_mask):
        t = cell_mask.time.values
        m = cell_mask.values
        dts = np.where(m == 1, t, np.datetime64('NaT'))
        dts = sorted(dts[~np.isnat(dts)])

        if len(dts) == 0:
            return None
        else:
            groups = [[dts[0]]]
            for i in dts:
                if i == groups[-1][-1] + np.timedelta64(86400, 's'):
                    groups[-1].append(i)
                else:
                    groups.append([i])
            groups = [v for v in groups if len(v) > 1]
            if len(groups) == 0:
                return None
            else:
                ds_groups = []
                for group in groups:
                    ts = pd.to_datetime(min(group)).replace(hour=0, minute=0, second=0)
                    te = pd.to_datetime(max(group)).replace(hour=23, minute=59, second=59)

                    _ds = xr.Dataset()
                    _ds = _ds.expand_dims({'event_id': [ts]})
                    _ds['event_start'] = (['event_id'], [ts])
                    _ds['event_end'] = (['event_id'], [te])
                    _ds['event_duration'] = np.ceil((_ds.event_end - _ds.event_start).astype(float) / (1e9 * 86400)).astype(int)
                    ds_groups.append(_ds)
                ds = xr.combine_by_coords(ds_groups)
                return ds

    def group_from_mask(self,mhw_mask):
        coords = [v for v in mhw_mask.coords if
                  v not in ['time', 'dayofyear']]  # Remove time-based coordinates for iterator.
        cell_groups = []
        for c in product(*[mhw_mask[coord] for coord in coords]):
            cell_mhw_mask = mhw_mask.sel({k: v for k, v in zip(coords, c)})
            grouped = self.group_cell_events(cell_mhw_mask)
            if grouped is not None:
                grouped = grouped.expand_dims({k: [v] for k, v in zip(coords, c)})
                grouped = grouped.sortby('event_id')
                cell_groups.append(grouped)
        gds = xr.concat(cell_groups, dim='event_id')
        gds = gds.sortby('event_id')
        return gds

    def group_stats(self,da, cda, pda, mhw_groups):
        coords = [v for v in mhw_groups.coords if v not in ['time', 'dayofyear', 'event_id']]
        new_cell_groups = []
        for c in product(*[mhw_groups[coord] for coord in coords]):
            cell_groups = mhw_groups.sel({k: v for k, v in zip(coords, c)})
            cell_groups = cell_groups.where(~np.isnan(cell_groups.event_duration), drop=True)
            if len(cell_groups.event_id) == 0: continue
            for event_id in cell_groups.event_id:
                cell_group = cell_groups.sel(event_id=event_id)

                da_sel = {k: [v] for k, v in zip(coords, c)}
                da_sel['time'] = slice(pd.to_datetime(cell_group.event_start.values),pd.to_datetime(cell_group.event_end.values))
                del da_sel['quantile']
                cell_da = da.sel(da_sel)

                # Subset the daily temperature climatology dataset based on unique selection criteria.
                cda_sel = {k: [v] for k, v in zip(coords, c)}
                del cda_sel['quantile']
                cda_sel['dayofyear'] = cell_da.time.dt.dayofyear
                cell_cda = cda.sel(cda_sel)

                # Subset the daily temperature threshold dataset based on unique selection criteria.
                pda_sel = {k: [v] for k, v in zip(coords, c)}
                pda_sel['dayofyear'] = cell_da.time.dt.dayofyear
                cell_pda = pda.sel(pda_sel)

                anom = cell_da - cell_cda

                # Calculate MHW time statistics.
                maxmask = xr.where(cell_da == cell_da.max(), 1, 0)
                temp_peak_time = maxmask.where(maxmask == 1, drop=True).time.values[0]
                cell_group[f'event_peak_{da.name}'] = cell_da.max()
                cell_group[f'event_peak_{da.name}_time'] = temp_peak_time

                maxmask = xr.where(anom == anom.max(), 1, 0)
                anom_peak_time = maxmask.where(maxmask == 1, drop=True).time.values[0]
                cell_group[f'event_peak_anomaly'] = anom.max()
                cell_group['event_peak_anomaly_time'] = anom_peak_time

                cell_group = cell_group.expand_dims(coords)  # I don't know why this only works here.

                # Calculate climatology MHW statistics.
                cell_group['event_intensity_max'] = anom.max(dim='time')
                cell_group['event_intensity_mean'] = anom.mean(dim='time')
                cell_group['event_intensity_stdev'] = anom.std(dim='time')
                cell_group['event_intensity_cumulative'] = anom.sum(dim='time')

                # Calculate category statistics.
                cat = np.floor(cell_group.event_intensity_max / (cell_pda - cell_cda))
                cat = cat.where(cat < 4, 4)
                cell_group['event_peak_category'] = cat.max(dim='time')

                cell_group = cell_group.expand_dims('event_id')
                new_cell_groups.append(cell_group)
        mhw_stats = xr.combine_by_coords(new_cell_groups)
        return mhw_stats
