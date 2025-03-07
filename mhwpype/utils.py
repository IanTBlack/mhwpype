import xarray as xr


def assign_depth(ds: xr.Dataset, depth: float) -> xr.Dataset:
    """
    This function assigns a depth coordinated to a 1D time-series dataset if that information
    is not available in the dataset. The depth input must be in meters.

    :param ds: The input dataset.
    :param depth: The depth value to be assigned. For example, sea surface temperature could have a depth of 0 meters.
    :return: The dataset with a depth coordinate.
    """
    ds = ds.expand_dims({'depth': [depth]})
    ds['depth'].attrs['short_name'] = 'depth'
    ds['depth'].attrs['long_name'] = 'depth'
    ds['depth'].attrs['units'] = 'meters'
    ds['depth'].attrs['units_abbreviation'] = 'm'
    ds['depth'].attrs['positive_direction'] = 'down'
    ds['depth'].attrs['axis'] = 'Z'
    return ds



def assign_location(ds: xr.Dataset, latitude: float, longitude: float) -> xr.Dataset:
    """
    This function assigns a latitude and longitude coordinate to a 1D time-series dataset if that information
    is not already available in the dataset. Latitude and longitude inputs must be in decimal degrees. It is recommended
    that latitude values fall between -90 and 90 and longitude values fall between -180 and 180.

    :param ds: The input dataset.
    :param latitude: A singular latitude value.
    :param longitude: A singular longitude value.
    :return: The dataset with latitude and longitude coordinates.
    """

    ds = ds.expand_dims({'latitude': [latitude], 'longitude': [longitude]})

    return ds


def reformat_longitude(ds: xr.Dataset) -> xr.Dataset:
    ds['longitude'] = ((ds.longitude + 180) % 360) - 180  # Convert longitude from [0 to 360] to [-180 to 180].
    ds = ds.sortby(['time', 'latitude', 'longitude'])
    ds['longitude'].attrs['units'] = 'degrees_east'
    ds['longitude'].attrs['units_abbreviation'] = 'E'
    ds['longitude'].attrs['range'] = [-180,180]
    ds['longitude'].attrs['actual_range'] = [float(ds['longitude'].min()), float(ds['longitude'].max())]
    return ds


def update_names(ds: xr.Dataset,
                 coord_mapper: dict = {'lat': 'latitude','lon': 'longitude'},
                 var_mapper: dict = {'sst': 'sea_water_temperature'}) -> xr.Dataset:
    for old_coord, new_coord in coord_mapper.items():
        if old_coord in ds.coords:
            ds = ds.rename({old_coord: new_coord})
    for old_var, new_var in var_mapper.items():
        if old_var in ds.data_vars:
            ds = ds.rename({old_var: new_var})
    return ds
