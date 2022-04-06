from collections import namedtuple

from pyomo.core.base.componentuid import ComponentUID
from pyomo.util.slices import slice_component_along_sets
from pyomo.core.base.indexed_component_slice import IndexedComponent_slice

from nmpc_examples.nmpc.dynamic_data.find_nearest_index import (
    find_nearest_index,
)


def get_time_indexed_cuid(var, sets=None, dereference=None):
    """
    Attempts to convert the provided "var" object into a CUID with
    with wildcards.

    Arguments
    ---------
    var:
        Object to process
    time: Set
        Set to use if slicing a vardata object
    dereference: None or int
        Number of times we may access referent attribute to recover a
        "base component" from a reference.

    """
    # TODO: Does this function have a good name?
    # Should this function be generalized beyond a single indexing set?
    if isinstance(var, ComponentUID):
        return var
    elif isinstance(var, (str, IndexedComponent_slice)):
        return ComponentUID(var)
    # At this point we are assuming var is a Pyomo Var or VarData object.

    # Is allowing dereference to be an integer worth the confusion it might
    # add?
    if dereference is None:
        # Does this branch make sense? If given an unattached component,
        # we dereference, otherwise we don't dereference.
        remaining_dereferences = int(var.parent_block() is None)
    else:
        remaining_dereferences = int(dereference)
    if var.is_indexed():
        if var.is_reference() and remaining_dereferences:
            remaining_dereferences -= 1
            referent = var.referent
            if isinstance(referent, IndexedComponent_slice):
                return ComponentUID(referent)
            else:
                # If dereference is None, we propagate None, dereferencing
                # until we either reach a component attached to a block
                # or reach a non-reference component.
                dereference = dereference if dereference is None else\
                        remaining_dereferences
                # NOTE: Calling this function recursively
                return get_time_indexed_cuid(
                    referent, time, dereference=dereference
                )
        else:
            # Assume that var is indexed only by time
            # TODO: Should we call slice_component_along_sets here as well?
            # To cover the case of b[t0].var, where var is indexed
            # by a set we care about, and we also care about time...
            # But then maybe we should slice only the sets we care about...
            # Don't want to do anything with these sets unless we're
            # presented with a vardata...
            index = tuple(
                get_slice_for_set(s) for s in var.index_set().subsets()
            )
            return ComponentUID(var[index])
    else:
        if sets is None:
            raise ValueError(
                "A ComponentData %s was provided but no set. We need to know\n"
                "what set this component should be indexed by."
                % var.name
            )
        slice_ = slice_component_along_sets(var, sets)
        return ComponentUID(slice_)


TimeSeriesTuple = namedtuple("TimeSeriesTuple", ["data", "time"])


class TimeSeriesData(object):
    """
    An object to store time series data associated with time-indexed
    variables.
    """

    def __init__(self, data, time, time_set=None):
        """
        Arguments:
        ----------
        data: dict or ComponentMap
            Maps variables, names, or CUIDs to lists of values
        time: list
            Contains the time points corresponding to variable data points.
        """
        # This is used if we ever need to process a VarData to get
        # a time-indexed CUID. We need to know what set to slice.
        self._orig_time_set = time_set
        self._time = list(time)
        self._time_idx_map = {t: idx for idx, t in enumerate(time)}

        if time is not None:
            # First make sure provided lists of variable data have the
            # same lengths as the provided time list.
            for key, data_list in data.items():
                # What if time is a number. Do I ever want to support that
                # here?
                if len(data_list) != len(time):
                    raise ValueError(
                        "Data lists must have same length as time. "
                        "Length of time is %s while length of data for "
                        "key %s is %s."
                        % (len(time), key, len(data_list))
                    )

        # Process keys of the provided data object to get CUIDs
        self._data = {
            get_time_indexed_cuid(key, (self._orig_time_set,)): values
            for key, values in data.items()
        }

    def get_time_points(self):
        """
        Get time points of the time series data
        """
        return self._time

    def get_data(self):
        """
        Return a dictionary mapping CUIDs to values
        """
        return self._data

    def get_data_at_time_indices(self, indices):
        """
        Returns data at the specified index or indices of this object's list
        of time points.
        """
        try:
            # Probably should raise an error if indices are not already sorted.
            index_list = list(sorted(indices))
            time_list = [self._time[i] for i in indices]
            data = {
                cuid: [values[idx] for idx in index_list]
                for cuid, values in self._data.items()
            }
            time_set = self._orig_time_set
            return TimeSeriesData(data, time_list, time_set=time_set)
        except TypeError:
            # indices is a scalar
            return {
                cuid: values[indices] for cuid, values in self._data.items()
            }

    def get_data_at_time(self, time=None, tolerance=None):
        """
        Returns the data associated with the provided time point or points.
        This function attempts to map time points to indices, then uses
        get_data_at_time_indices to actually extract the data. If a provided
        time point does not exist in the time-index map, binary search is
        used to find the closest value within a tolerance.

        Parameters
        ----------
        time: Float or iterable
            The time point or points corresponding to returned data.
        tolerance: Float
            Tolerance within which we will search for a matching time point.
            The default is None, which corresponds to an infinite tolerance.

        Returns
        -------
        TimeSeriesData or dict
            TimeSeriesData containing only the specified time points
            or dict mapping CUIDs to values at the specified scalar time
            point.

        """
        if time is None:
            # If time is not specified, assume we want the entire time
            # set. Skip all the overhead, don't create a new object, and
            # return self.
            return self
        try:
            indices = [
                self._time_idx_map[t] if t in self._time_idx_map else
                find_nearest_index(self._time, t, tolerance=tolerance)
                for t in time
            ]
        except TypeError:
            # time is a scalar
            indices = self._time_idx_map[time] if time in self._time_idx_map \
                else find_nearest_index(self._time, time, tolerance=tolerance)
        return self.get_data_at_time_indices(indices)

    def to_serializable(self):
        """
        Convert to json-serializable object.
        """
        time = self._time
        data = {str(cuid): values for cuid, values in self._data.items()}
        return TimeSeriesTuple(data, time)

    def concatenate(self, other):
        """
        Extend time list and variable data lists with the time points
        and variable values in the provided TimeSeriesData
        """
        # TODO: Potentially check here for "incompatible" time points,
        # i.e. violating sorted order. We don't assume that anywhere yet,
        # but it may be convenient to eventually.
        time = self._time.extend(other.get_time_points())
        
        data = self._data
        other_data = other.get_data()
        for cuid, values in data.items():
            # We assume that other contains all the cuids in self.
            # We make no assumption the other way around.
            values.extend(other_data[cuid])

    def shift_time_points(self, offset):
        """
        Apply an offset to stored time points.
        """
        # Note that this is different from what we are doing in
        # shift_values_by_time in the helper class.
        self._time = [t + offset for t in self._time]

    #
    # Unused
    #
    #def get_projection_onto_variables(self, variables):
    #    new = self.copy()
    #    new.project_onto_variables(variables)
    #    return new

    def extract_variables(self, variables):
        """
        Only keep variables specified by the user.
        """
        data = {}
        for var in variables:        
            cuid = get_time_indexed_cuid(var, (self._orig_time_set,))
            data[cuid] = self._data[cuid]
        return TimeSeriesData(
            data,
            self._time,
            time_set=self._orig_time_set,
        )

    #
    # Unused
    #
    #def copy(self):
    #    data = {key: list(values) for key, values in self._data.items()}
    #    time = list(self._time)
    #    return TimeSeriesData(data, time)
