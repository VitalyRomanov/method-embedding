gcrp_test_string = '''# Copyright 2019 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Interceptors implementation of gRPC Asyncio Python."""
import asyncio
import collections
import functools
from abc import ABCMeta, abstractmethod
from typing import Callable, Optional, Iterator, Sequence, Union, Awaitable
import grpc
from grpc._cython import cygrpc
from . import _base_call
from ._call import UnaryUnaryCall, AioRpcError
from ._utils import _timeout_to_deadline
from ._typing import (RequestType, SerializingFunction, DeserializingFunction,
                      MetadataType, ResponseType, DoneCallbackType)
_LOCAL_CANCELLATION_DETAILS = 'Locally cancelled by application!'
class ServerInterceptor(metaclass=ABCMeta):
    """Affords intercepting incoming RPCs on the service-side.
    This is an EXPERIMENTAL API.
    """
    @abstractmethod
    async def intercept_service(
            self, continuation: Callable[[grpc.HandlerCallDetails], Awaitable[
                grpc.RpcMethodHandler]],
            handler_call_details: grpc.HandlerCallDetails
    ) -> grpc.RpcMethodHandler:
        """Intercepts incoming RPCs before handing them over to a handler.
        Args:
            continuation: A function that takes a HandlerCallDetails and
                proceeds to invoke the next interceptor in the chain, if any,
                or the RPC handler lookup logic, with the call details passed
                as an argument, and returns an RpcMethodHandler instance if
                the RPC is considered serviced, or None otherwise.
            handler_call_details: A HandlerCallDetails describing the RPC.
        Returns:
            An RpcMethodHandler with which the RPC may be serviced if the
            interceptor chooses to service this RPC, or None otherwise.
        """
class ClientCallDetails(
        collections.namedtuple(
            'ClientCallDetails',
            ('method', 'timeout', 'metadata', 'credentials', 'wait_for_ready')),
        grpc.ClientCallDetails):
    """Describes an RPC to be invoked.
    This is an EXPERIMENTAL API.
    Args:
        method: The method name of the RPC.
        timeout: An optional duration of time in seconds to allow for the RPC.
        metadata: Optional metadata to be transmitted to the service-side of
          the RPC.
        credentials: An optional CallCredentials for the RPC.
        wait_for_ready: This is an EXPERIMENTAL argument. An optional flag to
          enable wait for ready mechanism.
    """
    method: str
    timeout: Optional[float]
    metadata: Optional[MetadataType]
    credentials: Optional[grpc.CallCredentials]
    wait_for_ready: Optional[bool]
class UnaryUnaryClientInterceptor(metaclass=ABCMeta):
    """Affords intercepting unary-unary invocations."""
    @abstractmethod
    async def intercept_unary_unary(
            self, continuation: Callable[[ClientCallDetails, RequestType],
                                         UnaryUnaryCall],
            client_call_details: ClientCallDetails,
            request: RequestType) -> Union[UnaryUnaryCall, ResponseType]:
        """Intercepts a unary-unary invocation asynchronously.
        Args:
          continuation: A coroutine that proceeds with the invocation by
            executing the next interceptor in chain or invoking the
            actual RPC on the underlying Channel. It is the interceptor's
            responsibility to call it if it decides to move the RPC forward.
            The interceptor can use
            `response_future = await continuation(client_call_details, request)`
            to continue with the RPC. `continuation` returns the response of the
            RPC.
          client_call_details: A ClientCallDetails object describing the
            outgoing RPC.
          request: The request value for the RPC.
        Returns:
          An object with the RPC response.
        Raises:
          AioRpcError: Indicating that the RPC terminated with non-OK status.
          asyncio.CancelledError: Indicating that the RPC was canceled.
        """
class InterceptedUnaryUnaryCall(_base_call.UnaryUnaryCall):
    """Used for running a `UnaryUnaryCall` wrapped by interceptors.
    Interceptors might have some work to do before the RPC invocation with
    the capacity of changing the invocation parameters, and some work to do
    after the RPC invocation with the capacity for accessing to the wrapped
    `UnaryUnaryCall`.
    It handles also early and later cancellations, when the RPC has not even
    started and the execution is still held by the interceptors or when the
    RPC has finished but again the execution is still held by the interceptors.
    Once the RPC is finally executed, all methods are finally done against the
    intercepted call, being at the same time the same call returned to the
    interceptors.
    For most of the methods, like `initial_metadata()` the caller does not need
    to wait until the interceptors task is finished, once the RPC is done the
    caller will have the freedom for accessing to the results.
    For the `__await__` method is it is proxied to the intercepted call only when
    the interceptor task is finished.
    """
    _loop: asyncio.AbstractEventLoop
    _channel: cygrpc.AioChannel
    _cancelled_before_rpc: bool
    _intercepted_call: Optional[_base_call.UnaryUnaryCall]
    _intercepted_call_created: asyncio.Event
    _interceptors_task: asyncio.Task
    _pending_add_done_callbacks: Sequence[DoneCallbackType]
    # pylint: disable=too-many-arguments
    def __init__(self, interceptors: Sequence[UnaryUnaryClientInterceptor],
                 request: RequestType, timeout: Optional[float],
                 metadata: MetadataType,
                 credentials: Optional[grpc.CallCredentials],
                 wait_for_ready: Optional[bool], channel: cygrpc.AioChannel,
                 method: bytes, request_serializer: SerializingFunction,
                 response_deserializer: DeserializingFunction,
                 loop: asyncio.AbstractEventLoop) -> None:
        self._channel = channel
        self._loop = loop
        self._interceptors_task = loop.create_task(
            self._invoke(interceptors, method, timeout, metadata, credentials,
                         wait_for_ready, request, request_serializer,
                         response_deserializer))
        self._pending_add_done_callbacks = []
        self._interceptors_task.add_done_callback(
            self._fire_pending_add_done_callbacks)
    def __del__(self):
        self.cancel()
    # pylint: disable=too-many-arguments
    async def _invoke(self, interceptors: Sequence[UnaryUnaryClientInterceptor],
                      method: bytes, timeout: Optional[float],
                      metadata: Optional[MetadataType],
                      credentials: Optional[grpc.CallCredentials],
                      wait_for_ready: Optional[bool], request: RequestType,
                      request_serializer: SerializingFunction,
                      response_deserializer: DeserializingFunction
                     ) -> UnaryUnaryCall:
        """Run the RPC call wrapped in interceptors"""
        async def _run_interceptor(
                interceptors: Iterator[UnaryUnaryClientInterceptor],
                client_call_details: ClientCallDetails,
                request: RequestType) -> _base_call.UnaryUnaryCall:
            interceptor = next(interceptors, None)
            if interceptor:
                continuation = functools.partial(_run_interceptor, interceptors)
                call_or_response = await interceptor.intercept_unary_unary(
                    continuation, client_call_details, request)
                if isinstance(call_or_response, _base_call.UnaryUnaryCall):
                    return call_or_response
                else:
                    return UnaryUnaryCallResponse(call_or_response)
            else:
                return UnaryUnaryCall(
                    request, _timeout_to_deadline(client_call_details.timeout),
                    client_call_details.metadata,
                    client_call_details.credentials,
                    client_call_details.wait_for_ready, self._channel,
                    client_call_details.method, request_serializer,
                    response_deserializer, self._loop)
        client_call_details = ClientCallDetails(method, timeout, metadata,
                                                credentials, wait_for_ready)
        return await _run_interceptor(iter(interceptors), client_call_details,
                                      request)
    def _fire_pending_add_done_callbacks(self,
                                         unused_task: asyncio.Task) -> None:
        for callback in self._pending_add_done_callbacks:
            callback(self)
        self._pending_add_done_callbacks = []
    def _wrap_add_done_callback(self, callback: DoneCallbackType,
                                unused_task: asyncio.Task) -> None:
        callback(self)
    def cancel(self) -> bool:
        if self._interceptors_task.done():
            return False
        return self._interceptors_task.cancel()
    def cancelled(self) -> bool:
        if not self._interceptors_task.done():
            return False
        try:
            call = self._interceptors_task.result()
        except AioRpcError as err:
            return err.code() == grpc.StatusCode.CANCELLED
        except asyncio.CancelledError:
            return True
        return call.cancelled()
    def done(self) -> bool:
        if not self._interceptors_task.done():
            return False
        try:
            call = self._interceptors_task.result()
        except (AioRpcError, asyncio.CancelledError):
            return True
        return call.done()
    def add_done_callback(self, callback: DoneCallbackType) -> None:
        if not self._interceptors_task.done():
            self._pending_add_done_callbacks.append(callback)
            return
        try:
            call = self._interceptors_task.result()
        except (AioRpcError, asyncio.CancelledError):
            callback(self)
            return
        if call.done():
            callback(self)
        else:
            callback = functools.partial(self._wrap_add_done_callback, callback)
            call.add_done_callback(self._wrap_add_done_callback)
    def time_remaining(self) -> Optional[float]:
        raise NotImplementedError()
    async def initial_metadata(self) -> Optional[MetadataType]:
        try:
            call = await self._interceptors_task
        except AioRpcError as err:
            return err.initial_metadata()
        except asyncio.CancelledError:
            return None
        return await call.initial_metadata()
    async def trailing_metadata(self) -> Optional[MetadataType]:
        try:
            call = await self._interceptors_task
        except AioRpcError as err:
            return err.trailing_metadata()
        except asyncio.CancelledError:
            return None
        return await call.trailing_metadata()
    async def code(self) -> grpc.StatusCode:
        try:
            call = await self._interceptors_task
        except AioRpcError as err:
            return err.code()
        except asyncio.CancelledError:
            return grpc.StatusCode.CANCELLED
        return await call.code()
    async def details(self) -> str:
        try:
            call = await self._interceptors_task
        except AioRpcError as err:
            return err.details()
        except asyncio.CancelledError:
            return _LOCAL_CANCELLATION_DETAILS
        return await call.details()
    async def debug_error_string(self) -> Optional[str]:
        try:
            call = await self._interceptors_task
        except AioRpcError as err:
            return err.debug_error_string()
        except asyncio.CancelledError:
            return ''
        return await call.debug_error_string()
    def __await__(self):
        call = yield from self._interceptors_task.__await__()
        response = yield from call.__await__()
        return response
    async def wait_for_connection(self) -> None:
        call = await self._interceptors_task
        return await call.wait_for_connection()
class UnaryUnaryCallResponse(_base_call.UnaryUnaryCall):
    """Final UnaryUnaryCall class finished with a response."""
    _response: ResponseType
    def __init__(self, response: ResponseType) -> None:
        self._response = response
    def cancel(self) -> bool:
        return False
    def cancelled(self) -> bool:
        return False
    def done(self) -> bool:
        return True
    def add_done_callback(self, unused_callback) -> None:
        raise NotImplementedError()
    def time_remaining(self) -> Optional[float]:
        raise NotImplementedError()
    async def initial_metadata(self) -> Optional[MetadataType]:
        return None
    async def trailing_metadata(self) -> Optional[MetadataType]:
        return None
    async def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.OK
    async def details(self) -> str:
        return ''
    async def debug_error_string(self) -> Optional[str]:
        return None
    def __await__(self):
        if False:  # pylint: disable=using-constant-test
            # This code path is never used, but a yield statement is needed
            # for telling the interpreter that __await__ is a generator.
            yield None
        return self._response
    async def wait_for_connection(self) -> None:
        pass
'''

gcrp_offsets = [{'start': 820, 'end': 824, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 1422, 'end': 1426, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 1475, 'end': 1479, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 1534, 'end': 1538, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 1567, 'end': 1571, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 2484, 'end': 2488, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 3112, 'end': 3116, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 6276, 'end': 6280, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 7359, 'end': 7363, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 9946, 'end': 9950, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 11661, 'end': 11665, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 11863, 'end': 11867, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 13601, 'end': 13605, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 13633, 'end': 13637, 'node_id': 2, 'occ_type': 2, 'name': 'grpc'}, {'start': 5762, 'end': 5768, 'node_id': 15, 'occ_type': 2, 'name': 'grpc._cython.cygrpc'}, {'start': 6357, 'end': 6363, 'node_id': 15, 'occ_type': 2, 'name': 'grpc._cython.cygrpc'}, {'start': 9951, 'end': 9961, 'node_id': 53, 'occ_type': 2, 'name': 'grpc.StatusCode'}, {'start': 11868, 'end': 11878, 'node_id': 53, 'occ_type': 2, 'name': 'grpc.StatusCode'}, {'start': 13638, 'end': 13648, 'node_id': 53, 'occ_type': 2, 'name': 'grpc.StatusCode'}, {'start': 2338, 'end': 2349, 'node_id': 194, 'occ_type': 2, 'name': 'collections'}, {'start': 4606, 'end': 4616, 'node_id': 1647, 'occ_type': 2, 'name': 'grpc.experimental.aio._base_call'}, {'start': 5844, 'end': 5854, 'node_id': 1647, 'occ_type': 2, 'name': 'grpc.experimental.aio._base_call'}, {'start': 7888, 'end': 7898, 'node_id': 1647, 'occ_type': 2, 'name': 'grpc.experimental.aio._base_call'}, {'start': 8267, 'end': 8277, 'node_id': 1647, 'occ_type': 2, 'name': 'grpc.experimental.aio._base_call'}, {'start': 12854, 'end': 12864, 'node_id': 1647, 'occ_type': 2, 'name': 'grpc.experimental.aio._base_call'}, {'start': 5722, 'end': 5729, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 5902, 'end': 5909, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 5940, 'end': 5947, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 6535, 'end': 6542, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 9245, 'end': 9252, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 9512, 'end': 9519, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 9987, 'end': 9994, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 10264, 'end': 10271, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 10634, 'end': 10641, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 11222, 'end': 11229, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 11537, 'end': 11544, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 11820, 'end': 11827, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 12105, 'end': 12112, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 12427, 'end': 12434, 'node_id': 1855, 'occ_type': 2, 'name': 'asyncio'}, {'start': 8027, 'end': 8036, 'node_id': 1861, 'occ_type': 2, 'name': 'functools'}, {'start': 10794, 'end': 10803, 'node_id': 1861, 'occ_type': 2, 'name': 'functools'}, {'start': 1109, 'end': 1136, 'node_id': 3958, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor._LOCAL_CANCELLATION_DETAILS'}, {'start': 1183, 'end': 1200, 'node_id': 3959, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ServerInterceptor'}, {'start': 1177, 'end': 2303, 'node_id': 3959, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.ServerInterceptor'}, {'start': 1361, 'end': 1378, 'node_id': 3960, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ServerInterceptor.intercept_service'}, {'start': 1357, 'end': 2303, 'node_id': 3960, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.ServerInterceptor.intercept_service'}, {'start': 2311, 'end': 2328, 'node_id': 3961, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails'}, {'start': 2305, 'end': 3169, 'node_id': 3961, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails'}, {'start': 3008, 'end': 3014, 'node_id': 3962, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails.method'}, {'start': 3024, 'end': 3031, 'node_id': 3963, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails.timeout'}, {'start': 3053, 'end': 3061, 'node_id': 3964, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails.metadata'}, {'start': 3090, 'end': 3101, 'node_id': 3965, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails.credentials'}, {'start': 3138, 'end': 3152, 'node_id': 3966, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails.wait_for_ready'}, {'start': 3177, 'end': 3204, 'node_id': 3967, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryClientInterceptor'}, {'start': 3171, 'end': 4572, 'node_id': 3967, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryClientInterceptor'}, {'start': 3316, 'end': 3337, 'node_id': 3968, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryClientInterceptor.intercept_unary_unary'}, {'start': 3312, 'end': 4572, 'node_id': 3968, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryClientInterceptor.intercept_unary_unary'}, {'start': 4580, 'end': 4605, 'node_id': 3969, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall'}, {'start': 4574, 'end': 12823, 'node_id': 3969, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall'}, {'start': 5715, 'end': 5720, 'node_id': 3970, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._loop'}, {'start': 6616, 'end': 6621, 'node_id': 3970, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._loop'}, {'start': 5752, 'end': 5760, 'node_id': 3971, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._channel'}, {'start': 6584, 'end': 6592, 'node_id': 3971, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._channel'}, {'start': 5784, 'end': 5805, 'node_id': 3972, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._cancelled_before_rpc'}, {'start': 5816, 'end': 5833, 'node_id': 3973, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._intercepted_call'}, {'start': 5875, 'end': 5900, 'node_id': 3974, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._intercepted_call_created'}, {'start': 5920, 'end': 5938, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 6642, 'end': 6660, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 5957, 'end': 5984, 'node_id': 3976, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._pending_add_done_callbacks'}, {'start': 6892, 'end': 6919, 'node_id': 3976, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._pending_add_done_callbacks'}, {'start': 9367, 'end': 9394, 'node_id': 3976, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._pending_add_done_callbacks'}, {'start': 6063, 'end': 6071, 'node_id': 3977, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.__init__'}, {'start': 6059, 'end': 7027, 'node_id': 3977, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.__init__'}, {'start': 7129, 'end': 7136, 'node_id': 3978, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._invoke'}, {'start': 7125, 'end': 9143, 'node_id': 3978, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._invoke'}, {'start': 9152, 'end': 9184, 'node_id': 3982, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._fire_pending_add_done_callbacks'}, {'start': 9148, 'end': 9400, 'node_id': 3982, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._fire_pending_add_done_callbacks'}, {'start': 7036, 'end': 7043, 'node_id': 3983, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.__del__'}, {'start': 7032, 'end': 7073, 'node_id': 3983, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.__del__'}, {'start': 9567, 'end': 9573, 'node_id': 3984, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.cancel'}, {'start': 9563, 'end': 9706, 'node_id': 3984, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.cancel'}, {'start': 7704, 'end': 7720, 'node_id': 3985, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._invoke._run_interceptor'}, {'start': 7700, 'end': 8864, 'node_id': 3985, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._invoke._run_interceptor'}, {'start': 12831, 'end': 12853, 'node_id': 3986, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse'}, {'start': 12825, 'end': 14133, 'node_id': 3986, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse'}, {'start': 12982, 'end': 12990, 'node_id': 3987, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.__init__'}, {'start': 12978, 'end': 13064, 'node_id': 3987, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.__init__'}, {'start': 9409, 'end': 9432, 'node_id': 3989, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._wrap_add_done_callback'}, {'start': 9405, 'end': 9558, 'node_id': 3989, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._wrap_add_done_callback'}, {'start': 9715, 'end': 9724, 'node_id': 3992, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.cancelled'}, {'start': 9711, 'end': 10068, 'node_id': 3992, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.cancelled'}, {'start': 10077, 'end': 10081, 'node_id': 3995, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.done'}, {'start': 10073, 'end': 10341, 'node_id': 3995, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.done'}, {'start': 10350, 'end': 10367, 'node_id': 3996, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.add_done_callback'}, {'start': 10346, 'end': 10917, 'node_id': 3996, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.add_done_callback'}, {'start': 10926, 'end': 10940, 'node_id': 3997, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.time_remaining'}, {'start': 10922, 'end': 11003, 'node_id': 3997, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.time_remaining'}, {'start': 11018, 'end': 11034, 'node_id': 3998, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.initial_metadata'}, {'start': 11014, 'end': 11316, 'node_id': 3998, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.initial_metadata'}, {'start': 11331, 'end': 11348, 'node_id': 3999, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.trailing_metadata'}, {'start': 11327, 'end': 11632, 'node_id': 3999, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.trailing_metadata'}, {'start': 11647, 'end': 11651, 'node_id': 4000, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.code'}, {'start': 11643, 'end': 11923, 'node_id': 4000, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.code'}, {'start': 11938, 'end': 11945, 'node_id': 4001, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.details'}, {'start': 11934, 'end': 12213, 'node_id': 4001, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.details'}, {'start': 12228, 'end': 12246, 'node_id': 4002, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.debug_error_string'}, {'start': 12224, 'end': 12521, 'node_id': 4002, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.debug_error_string'}, {'start': 12530, 'end': 12539, 'node_id': 4003, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.__await__'}, {'start': 12526, 'end': 12680, 'node_id': 4003, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.__await__'}, {'start': 12695, 'end': 12714, 'node_id': 4006, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.wait_for_connection'}, {'start': 12691, 'end': 12823, 'node_id': 4006, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.wait_for_connection'}, {'start': 12949, 'end': 12958, 'node_id': 4007, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse._response'}, {'start': 13043, 'end': 13052, 'node_id': 4007, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse._response'}, {'start': 13073, 'end': 13079, 'node_id': 4008, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.cancel'}, {'start': 13069, 'end': 13116, 'node_id': 4008, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.cancel'}, {'start': 13125, 'end': 13134, 'node_id': 4009, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.cancelled'}, {'start': 13121, 'end': 13171, 'node_id': 4009, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.cancelled'}, {'start': 13180, 'end': 13184, 'node_id': 4010, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.done'}, {'start': 13176, 'end': 13220, 'node_id': 4010, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.done'}, {'start': 13229, 'end': 13246, 'node_id': 4011, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.add_done_callback'}, {'start': 13225, 'end': 13315, 'node_id': 4011, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.add_done_callback'}, {'start': 13324, 'end': 13338, 'node_id': 4012, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.time_remaining'}, {'start': 13320, 'end': 13401, 'node_id': 4012, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.time_remaining'}, {'start': 13416, 'end': 13432, 'node_id': 4013, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.initial_metadata'}, {'start': 13412, 'end': 13486, 'node_id': 4013, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.initial_metadata'}, {'start': 13501, 'end': 13518, 'node_id': 4014, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.trailing_metadata'}, {'start': 13497, 'end': 13572, 'node_id': 4014, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.trailing_metadata'}, {'start': 13587, 'end': 13591, 'node_id': 4015, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.code'}, {'start': 13583, 'end': 13652, 'node_id': 4015, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.code'}, {'start': 13667, 'end': 13674, 'node_id': 4016, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.details'}, {'start': 13663, 'end': 13707, 'node_id': 4016, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.details'}, {'start': 13722, 'end': 13740, 'node_id': 4017, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.debug_error_string'}, {'start': 13718, 'end': 13785, 'node_id': 4017, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.debug_error_string'}, {'start': 13794, 'end': 13803, 'node_id': 4018, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.__await__'}, {'start': 13790, 'end': 14070, 'node_id': 4018, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.__await__'}, {'start': 14085, 'end': 14104, 'node_id': 4019, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.wait_for_connection'}, {'start': 14081, 'end': 14133, 'node_id': 4019, 'occ_type': 1, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.wait_for_connection'}, {'start': 642, 'end': 649, 'node_id': 1855, 'occ_type': 0, 'name': 'asyncio'}, {'start': 657, 'end': 668, 'node_id': 194, 'occ_type': 0, 'name': 'collections'}, {'start': 676, 'end': 685, 'node_id': 1861, 'occ_type': 0, 'name': 'functools'}, {'start': 691, 'end': 694, 'node_id': 1637, 'occ_type': 0, 'name': 'abc'}, {'start': 702, 'end': 709, 'node_id': 2870, 'occ_type': 0, 'name': 'abc.ABCMeta'}, {'start': 711, 'end': 725, 'node_id': 1674, 'occ_type': 0, 'name': 'abc.abstractmethod'}, {'start': 731, 'end': 737, 'node_id': 415, 'occ_type': 0, 'name': 'typing'}, {'start': 745, 'end': 753, 'node_id': 3343, 'occ_type': 0, 'name': 'typing.Callable'}, {'start': 755, 'end': 763, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 765, 'end': 773, 'node_id': 3704, 'occ_type': 0, 'name': 'typing.Iterator'}, {'start': 775, 'end': 783, 'node_id': 3349, 'occ_type': 0, 'name': 'typing.Sequence'}, {'start': 785, 'end': 790, 'node_id': 3356, 'occ_type': 0, 'name': 'typing.Union'}, {'start': 792, 'end': 801, 'node_id': 3955, 'occ_type': 0, 'name': 'typing.Awaitable'}, {'start': 810, 'end': 814, 'node_id': 2, 'occ_type': 0, 'name': 'grpc'}, {'start': 825, 'end': 832, 'node_id': 12, 'occ_type': 0, 'name': 'grpc._cython'}, {'start': 840, 'end': 846, 'node_id': 15, 'occ_type': 0, 'name': 'grpc._cython.cygrpc'}, {'start': 862, 'end': 872, 'node_id': 1647, 'occ_type': 0, 'name': 'grpc.experimental.aio._base_call'}, {'start': 879, 'end': 884, 'node_id': 1853, 'occ_type': 0, 'name': 'grpc.experimental.aio._call'}, {'start': 892, 'end': 906, 'node_id': 2530, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.UnaryUnaryCall'}, {'start': 908, 'end': 919, 'node_id': 1915, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError'}, {'start': 926, 'end': 932, 'node_id': 3956, 'occ_type': 0, 'name': 'grpc.experimental.aio._utils'}, {'start': 940, 'end': 960, 'node_id': 3957, 'occ_type': 0, 'name': 'grpc.experimental.aio._utils._timeout_to_deadline'}, {'start': 967, 'end': 974, 'node_id': 1650, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing'}, {'start': 983, 'end': 994, 'node_id': 1890, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.RequestType'}, {'start': 996, 'end': 1015, 'node_id': 1662, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.SerializingFunction'}, {'start': 1017, 'end': 1038, 'node_id': 1653, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.DeserializingFunction'}, {'start': 1062, 'end': 1074, 'node_id': 1656, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.MetadataType'}, {'start': 1076, 'end': 1088, 'node_id': 1893, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.ResponseType'}, {'start': 1090, 'end': 1106, 'node_id': 1882, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.DoneCallbackType'}, {'start': 1211, 'end': 1218, 'node_id': 2870, 'occ_type': 0, 'name': 'abc.ABCMeta'}, {'start': 1412, 'end': 1420, 'node_id': 3343, 'occ_type': 0, 'name': 'typing.Callable'}, {'start': 1427, 'end': 1445, 'node_id': 445, 'occ_type': 0, 'name': 'grpc.HandlerCallDetails'}, {'start': 1539, 'end': 1557, 'node_id': 445, 'occ_type': 0, 'name': 'grpc.HandlerCallDetails'}, {'start': 1448, 'end': 1457, 'node_id': 3955, 'occ_type': 0, 'name': 'typing.Awaitable'}, {'start': 1480, 'end': 1496, 'node_id': 3622, 'occ_type': 0, 'name': 'grpc.RpcMethodHandler'}, {'start': 1572, 'end': 1588, 'node_id': 3622, 'occ_type': 0, 'name': 'grpc.RpcMethodHandler'}, {'start': 2350, 'end': 2360, 'node_id': 210, 'occ_type': 0, 'name': 'collections.namedtuple'}, {'start': 2489, 'end': 2506, 'node_id': 3565, 'occ_type': 0, 'name': 'grpc.ClientCallDetails'}, {'start': 3016, 'end': 3019, 'node_id': 127, 'occ_type': 0, 'name': 'builtins.str'}, {'start': 3033, 'end': 3041, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 3063, 'end': 3071, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 3103, 'end': 3111, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 3154, 'end': 3162, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 3042, 'end': 3047, 'node_id': 1684, 'occ_type': 0, 'name': 'builtins.float'}, {'start': 3072, 'end': 3084, 'node_id': 1656, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.MetadataType'}, {'start': 3117, 'end': 3132, 'node_id': 331, 'occ_type': 0, 'name': 'grpc.CallCredentials'}, {'start': 3163, 'end': 3167, 'node_id': 1693, 'occ_type': 0, 'name': 'builtins.bool'}, {'start': 3215, 'end': 3222, 'node_id': 2870, 'occ_type': 0, 'name': 'abc.ABCMeta'}, {'start': 3371, 'end': 3379, 'node_id': 3343, 'occ_type': 0, 'name': 'typing.Callable'}, {'start': 3381, 'end': 3398, 'node_id': 3961, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails'}, {'start': 3505, 'end': 3522, 'node_id': 3961, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails'}, {'start': 3400, 'end': 3411, 'node_id': 1890, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.RequestType'}, {'start': 3545, 'end': 3556, 'node_id': 1890, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.RequestType'}, {'start': 3455, 'end': 3469, 'node_id': 2530, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.UnaryUnaryCall'}, {'start': 3567, 'end': 3581, 'node_id': 2530, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.UnaryUnaryCall'}, {'start': 3561, 'end': 3566, 'node_id': 3356, 'occ_type': 0, 'name': 'typing.Union'}, {'start': 3583, 'end': 3595, 'node_id': 1893, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.ResponseType'}, {'start': 4617, 'end': 4631, 'node_id': 1700, 'occ_type': 0, 'name': 'grpc.experimental.aio._base_call.UnaryUnaryCall'}, {'start': 5855, 'end': 5869, 'node_id': 1700, 'occ_type': 0, 'name': 'grpc.experimental.aio._base_call.UnaryUnaryCall'}, {'start': 5730, 'end': 5747, 'node_id': 2039, 'occ_type': 0, 'name': 'asyncio.events.AbstractEventLoop'}, {'start': 5769, 'end': 5779, 'node_id': 2557, 'occ_type': 0, 'name': 'grpc._cython.cygrpc.AioChannel'}, {'start': 5807, 'end': 5811, 'node_id': 1693, 'occ_type': 0, 'name': 'builtins.bool'}, {'start': 5835, 'end': 5843, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 5910, 'end': 5915, 'node_id': 2387, 'occ_type': 0, 'name': 'asyncio.locks.Event'}, {'start': 5948, 'end': 5952, 'node_id': 2241, 'occ_type': 0, 'name': 'asyncio.tasks.Task'}, {'start': 5948, 'end': 5952, 'node_id': 2241, 'occ_type': 0, 'name': 'asyncio.tasks.Task'}, {'start': 5986, 'end': 5994, 'node_id': 3349, 'occ_type': 0, 'name': 'typing.Sequence'}, {'start': 5995, 'end': 6011, 'node_id': 1882, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.DoneCallbackType'}, {'start': 6092, 'end': 6100, 'node_id': 3349, 'occ_type': 0, 'name': 'typing.Sequence'}, {'start': 6101, 'end': 6128, 'node_id': 3967, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryClientInterceptor'}, {'start': 6157, 'end': 6168, 'node_id': 1890, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.RequestType'}, {'start': 6179, 'end': 6187, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 6267, 'end': 6275, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 6332, 'end': 6340, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 6188, 'end': 6193, 'node_id': 1684, 'occ_type': 0, 'name': 'builtins.float'}, {'start': 6223, 'end': 6235, 'node_id': 1656, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.MetadataType'}, {'start': 6281, 'end': 6296, 'node_id': 331, 'occ_type': 0, 'name': 'grpc.CallCredentials'}, {'start': 6341, 'end': 6345, 'node_id': 1693, 'occ_type': 0, 'name': 'builtins.bool'}, {'start': 6364, 'end': 6374, 'node_id': 2557, 'occ_type': 0, 'name': 'grpc._cython.cygrpc.AioChannel'}, {'start': 6401, 'end': 6406, 'node_id': 124, 'occ_type': 0, 'name': 'builtins.bytes'}, {'start': 6428, 'end': 6447, 'node_id': 1662, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.SerializingFunction'}, {'start': 6489, 'end': 6510, 'node_id': 1653, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.DeserializingFunction'}, {'start': 6543, 'end': 6560, 'node_id': 2039, 'occ_type': 0, 'name': 'asyncio.events.AbstractEventLoop'}, {'start': 6584, 'end': 6592, 'node_id': 3971, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._channel'}, {'start': 6616, 'end': 6621, 'node_id': 3970, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._loop'}, {'start': 6642, 'end': 6660, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 6938, 'end': 6956, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 6668, 'end': 6679, 'node_id': 2416, 'occ_type': 0, 'name': 'asyncio.events.AbstractEventLoop.create_task'}, {'start': 6698, 'end': 6705, 'node_id': 3978, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._invoke'}, {'start': 6892, 'end': 6919, 'node_id': 3976, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._pending_add_done_callbacks'}, {'start': 6957, 'end': 6974, 'node_id': 3979, 'occ_type': 0, 'name': '_asyncio.add_done_callback'}, {'start': 6957, 'end': 6974, 'node_id': 3981, 'occ_type': 0, 'name': 'asyncio.futures.Future.add_done_callback'}, {'start': 7064, 'end': 7070, 'node_id': 3984, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall.cancel'}, {'start': 7157, 'end': 7165, 'node_id': 3349, 'occ_type': 0, 'name': 'typing.Sequence'}, {'start': 7166, 'end': 7193, 'node_id': 3967, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryClientInterceptor'}, {'start': 7226, 'end': 7231, 'node_id': 124, 'occ_type': 0, 'name': 'builtins.bytes'}, {'start': 7242, 'end': 7250, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 7291, 'end': 7299, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 7350, 'end': 7358, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 7420, 'end': 7428, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 7251, 'end': 7256, 'node_id': 1684, 'occ_type': 0, 'name': 'builtins.float'}, {'start': 7300, 'end': 7312, 'node_id': 1656, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.MetadataType'}, {'start': 7364, 'end': 7379, 'node_id': 331, 'occ_type': 0, 'name': 'grpc.CallCredentials'}, {'start': 7429, 'end': 7433, 'node_id': 1693, 'occ_type': 0, 'name': 'builtins.bool'}, {'start': 7445, 'end': 7456, 'node_id': 1890, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.RequestType'}, {'start': 7500, 'end': 7519, 'node_id': 1662, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.SerializingFunction'}, {'start': 7566, 'end': 7587, 'node_id': 1653, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.DeserializingFunction'}, {'start': 7614, 'end': 7628, 'node_id': 2530, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.UnaryUnaryCall'}, {'start': 7752, 'end': 7760, 'node_id': 3704, 'occ_type': 0, 'name': 'typing.Iterator'}, {'start': 7761, 'end': 7788, 'node_id': 3967, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryClientInterceptor'}, {'start': 7828, 'end': 7845, 'node_id': 3961, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails'}, {'start': 7872, 'end': 7883, 'node_id': 1890, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.RequestType'}, {'start': 7899, 'end': 7913, 'node_id': 1700, 'occ_type': 0, 'name': 'grpc.experimental.aio._base_call.UnaryUnaryCall'}, {'start': 8278, 'end': 8292, 'node_id': 1700, 'occ_type': 0, 'name': 'grpc.experimental.aio._base_call.UnaryUnaryCall'}, {'start': 7942, 'end': 7946, 'node_id': 975, 'occ_type': 0, 'name': 'builtins.next'}, {'start': 8037, 'end': 8044, 'node_id': 1863, 'occ_type': 0, 'name': 'functools.partial'}, {'start': 8037, 'end': 8044, 'node_id': 2128, 'occ_type': 0, 'name': 'functools.partial.__init__'}, {'start': 8131, 'end': 8152, 'node_id': 3968, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryClientInterceptor.intercept_unary_unary'}, {'start': 8238, 'end': 8248, 'node_id': 121, 'occ_type': 0, 'name': 'builtins.isinstance'}, {'start': 8388, 'end': 8410, 'node_id': 3986, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse'}, {'start': 8388, 'end': 8410, 'node_id': 3987, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse.__init__'}, {'start': 8471, 'end': 8485, 'node_id': 2530, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.UnaryUnaryCall'}, {'start': 8471, 'end': 8485, 'node_id': 2542, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.UnaryUnaryCall.__init__'}, {'start': 8516, 'end': 8536, 'node_id': 3957, 'occ_type': 0, 'name': 'grpc.experimental.aio._utils._timeout_to_deadline'}, {'start': 8557, 'end': 8564, 'node_id': 3963, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails.timeout'}, {'start': 8607, 'end': 8615, 'node_id': 3964, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails.metadata'}, {'start': 8657, 'end': 8668, 'node_id': 3965, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails.credentials'}, {'start': 8710, 'end': 8724, 'node_id': 3966, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails.wait_for_ready'}, {'start': 8731, 'end': 8739, 'node_id': 3971, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._channel'}, {'start': 8781, 'end': 8787, 'node_id': 3962, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails.method'}, {'start': 8857, 'end': 8862, 'node_id': 3970, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._loop'}, {'start': 8895, 'end': 8912, 'node_id': 3961, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails'}, {'start': 8895, 'end': 8912, 'node_id': 3988, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.ClientCallDetails.__init__'}, {'start': 9038, 'end': 9054, 'node_id': 3985, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._invoke._run_interceptor'}, {'start': 9055, 'end': 9059, 'node_id': 3740, 'occ_type': 0, 'name': 'builtins.iter'}, {'start': 9253, 'end': 9257, 'node_id': 2241, 'occ_type': 0, 'name': 'asyncio.tasks.Task'}, {'start': 9253, 'end': 9257, 'node_id': 2241, 'occ_type': 0, 'name': 'asyncio.tasks.Task'}, {'start': 9367, 'end': 9394, 'node_id': 3976, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._pending_add_done_callbacks'}, {'start': 9297, 'end': 9324, 'node_id': 3976, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._pending_add_done_callbacks'}, {'start': 9449, 'end': 9465, 'node_id': 1882, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.DoneCallbackType'}, {'start': 9520, 'end': 9524, 'node_id': 2241, 'occ_type': 0, 'name': 'asyncio.tasks.Task'}, {'start': 9520, 'end': 9524, 'node_id': 2241, 'occ_type': 0, 'name': 'asyncio.tasks.Task'}, {'start': 9583, 'end': 9587, 'node_id': 1693, 'occ_type': 0, 'name': 'builtins.bool'}, {'start': 9605, 'end': 9623, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 9678, 'end': 9696, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 9624, 'end': 9628, 'node_id': 3990, 'occ_type': 0, 'name': '_asyncio.done'}, {'start': 9624, 'end': 9628, 'node_id': 3991, 'occ_type': 0, 'name': 'asyncio.futures.Future.done'}, {'start': 9697, 'end': 9703, 'node_id': 2261, 'occ_type': 0, 'name': '_asyncio.cancel'}, {'start': 9697, 'end': 9703, 'node_id': 2264, 'occ_type': 0, 'name': 'asyncio.tasks.Task.cancel'}, {'start': 9734, 'end': 9738, 'node_id': 1693, 'occ_type': 0, 'name': 'builtins.bool'}, {'start': 9760, 'end': 9778, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 9850, 'end': 9868, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 9779, 'end': 9783, 'node_id': 3990, 'occ_type': 0, 'name': '_asyncio.done'}, {'start': 9779, 'end': 9783, 'node_id': 3991, 'occ_type': 0, 'name': 'asyncio.futures.Future.done'}, {'start': 9869, 'end': 9875, 'node_id': 3993, 'occ_type': 0, 'name': '_asyncio.result'}, {'start': 9869, 'end': 9875, 'node_id': 3994, 'occ_type': 0, 'name': 'asyncio.futures.Future.result'}, {'start': 9893, 'end': 9904, 'node_id': 1915, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError'}, {'start': 9936, 'end': 9940, 'node_id': 1960, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError.code'}, {'start': 9995, 'end': 10009, 'node_id': 2193, 'occ_type': 0, 'name': 'concurrent.futures._base.CancelledError'}, {'start': 10091, 'end': 10095, 'node_id': 1693, 'occ_type': 0, 'name': 'builtins.bool'}, {'start': 10117, 'end': 10135, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 10207, 'end': 10225, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 10136, 'end': 10140, 'node_id': 3990, 'occ_type': 0, 'name': '_asyncio.done'}, {'start': 10136, 'end': 10140, 'node_id': 3991, 'occ_type': 0, 'name': 'asyncio.futures.Future.done'}, {'start': 10226, 'end': 10232, 'node_id': 3993, 'occ_type': 0, 'name': '_asyncio.result'}, {'start': 10226, 'end': 10232, 'node_id': 3994, 'occ_type': 0, 'name': 'asyncio.futures.Future.result'}, {'start': 10251, 'end': 10262, 'node_id': 1915, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError'}, {'start': 10272, 'end': 10286, 'node_id': 2193, 'occ_type': 0, 'name': 'concurrent.futures._base.CancelledError'}, {'start': 10384, 'end': 10400, 'node_id': 1882, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.DoneCallbackType'}, {'start': 10431, 'end': 10449, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 10577, 'end': 10595, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 10450, 'end': 10454, 'node_id': 3990, 'occ_type': 0, 'name': '_asyncio.done'}, {'start': 10450, 'end': 10454, 'node_id': 3991, 'occ_type': 0, 'name': 'asyncio.futures.Future.done'}, {'start': 10475, 'end': 10502, 'node_id': 3976, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._pending_add_done_callbacks'}, {'start': 10503, 'end': 10509, 'node_id': 518, 'occ_type': 0, 'name': 'builtins.list.append'}, {'start': 10596, 'end': 10602, 'node_id': 3993, 'occ_type': 0, 'name': '_asyncio.result'}, {'start': 10596, 'end': 10602, 'node_id': 3994, 'occ_type': 0, 'name': 'asyncio.futures.Future.result'}, {'start': 10621, 'end': 10632, 'node_id': 1915, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError'}, {'start': 10642, 'end': 10656, 'node_id': 2193, 'occ_type': 0, 'name': 'concurrent.futures._base.CancelledError'}, {'start': 10804, 'end': 10811, 'node_id': 1863, 'occ_type': 0, 'name': 'functools.partial'}, {'start': 10804, 'end': 10811, 'node_id': 2128, 'occ_type': 0, 'name': 'functools.partial.__init__'}, {'start': 10950, 'end': 10958, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 10959, 'end': 10964, 'node_id': 1684, 'occ_type': 0, 'name': 'builtins.float'}, {'start': 10981, 'end': 11000, 'node_id': 2932, 'occ_type': 0, 'name': 'builtins.NotImplementedError'}, {'start': 10981, 'end': 11000, 'node_id': 2935, 'occ_type': 0, 'name': 'builtins.NotImplementedError.__init__'}, {'start': 11044, 'end': 11052, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 11053, 'end': 11065, 'node_id': 1656, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.MetadataType'}, {'start': 11111, 'end': 11129, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 11145, 'end': 11156, 'node_id': 1915, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError'}, {'start': 11188, 'end': 11204, 'node_id': 1971, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError.initial_metadata'}, {'start': 11230, 'end': 11244, 'node_id': 2193, 'occ_type': 0, 'name': 'concurrent.futures._base.CancelledError'}, {'start': 11358, 'end': 11366, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 11367, 'end': 11379, 'node_id': 1656, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.MetadataType'}, {'start': 11425, 'end': 11443, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 11459, 'end': 11470, 'node_id': 1915, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError'}, {'start': 11502, 'end': 11519, 'node_id': 1977, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError.trailing_metadata'}, {'start': 11545, 'end': 11559, 'node_id': 2193, 'occ_type': 0, 'name': 'concurrent.futures._base.CancelledError'}, {'start': 11666, 'end': 11676, 'node_id': 53, 'occ_type': 0, 'name': 'grpc.StatusCode'}, {'start': 11721, 'end': 11739, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 11755, 'end': 11766, 'node_id': 1915, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError'}, {'start': 11798, 'end': 11802, 'node_id': 1960, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError.code'}, {'start': 11828, 'end': 11842, 'node_id': 2193, 'occ_type': 0, 'name': 'concurrent.futures._base.CancelledError'}, {'start': 11955, 'end': 11958, 'node_id': 127, 'occ_type': 0, 'name': 'builtins.str'}, {'start': 12003, 'end': 12021, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 12037, 'end': 12048, 'node_id': 1915, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError'}, {'start': 12080, 'end': 12087, 'node_id': 1965, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError.details'}, {'start': 12113, 'end': 12127, 'node_id': 2193, 'occ_type': 0, 'name': 'concurrent.futures._base.CancelledError'}, {'start': 12148, 'end': 12175, 'node_id': 3958, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor._LOCAL_CANCELLATION_DETAILS'}, {'start': 12256, 'end': 12264, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 12265, 'end': 12268, 'node_id': 127, 'occ_type': 0, 'name': 'builtins.str'}, {'start': 12314, 'end': 12332, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 12348, 'end': 12359, 'node_id': 1915, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError'}, {'start': 12391, 'end': 12409, 'node_id': 1983, 'occ_type': 0, 'name': 'grpc.experimental.aio._call.AioRpcError.debug_error_string'}, {'start': 12435, 'end': 12449, 'node_id': 2193, 'occ_type': 0, 'name': 'concurrent.futures._base.CancelledError'}, {'start': 12578, 'end': 12596, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 12597, 'end': 12606, 'node_id': 4004, 'occ_type': 0, 'name': '_asyncio.__await__'}, {'start': 12597, 'end': 12606, 'node_id': 4005, 'occ_type': 0, 'name': 'asyncio.futures.Future.__await__'}, {'start': 12756, 'end': 12774, 'node_id': 3975, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.InterceptedUnaryUnaryCall._interceptors_task'}, {'start': 12865, 'end': 12879, 'node_id': 1700, 'occ_type': 0, 'name': 'grpc.experimental.aio._base_call.UnaryUnaryCall'}, {'start': 12960, 'end': 12972, 'node_id': 1893, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.ResponseType'}, {'start': 13007, 'end': 13019, 'node_id': 1893, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.ResponseType'}, {'start': 13043, 'end': 13052, 'node_id': 4007, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse._response'}, {'start': 13089, 'end': 13093, 'node_id': 1693, 'occ_type': 0, 'name': 'builtins.bool'}, {'start': 13144, 'end': 13148, 'node_id': 1693, 'occ_type': 0, 'name': 'builtins.bool'}, {'start': 13194, 'end': 13198, 'node_id': 1693, 'occ_type': 0, 'name': 'builtins.bool'}, {'start': 13293, 'end': 13312, 'node_id': 2932, 'occ_type': 0, 'name': 'builtins.NotImplementedError'}, {'start': 13293, 'end': 13312, 'node_id': 2935, 'occ_type': 0, 'name': 'builtins.NotImplementedError.__init__'}, {'start': 13348, 'end': 13356, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 13357, 'end': 13362, 'node_id': 1684, 'occ_type': 0, 'name': 'builtins.float'}, {'start': 13379, 'end': 13398, 'node_id': 2932, 'occ_type': 0, 'name': 'builtins.NotImplementedError'}, {'start': 13379, 'end': 13398, 'node_id': 2935, 'occ_type': 0, 'name': 'builtins.NotImplementedError.__init__'}, {'start': 13442, 'end': 13450, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 13451, 'end': 13463, 'node_id': 1656, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.MetadataType'}, {'start': 13528, 'end': 13536, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 13537, 'end': 13549, 'node_id': 1656, 'occ_type': 0, 'name': 'grpc.experimental.aio._typing.MetadataType'}, {'start': 13606, 'end': 13616, 'node_id': 53, 'occ_type': 0, 'name': 'grpc.StatusCode'}, {'start': 13684, 'end': 13687, 'node_id': 127, 'occ_type': 0, 'name': 'builtins.str'}, {'start': 13750, 'end': 13758, 'node_id': 1643, 'occ_type': 0, 'name': 'typing.Optional'}, {'start': 13759, 'end': 13762, 'node_id': 127, 'occ_type': 0, 'name': 'builtins.str'}, {'start': 14060, 'end': 14069, 'node_id': 4007, 'occ_type': 0, 'name': 'grpc.experimental.aio._interceptor.UnaryUnaryCallResponse._response'}]