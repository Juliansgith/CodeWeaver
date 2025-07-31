import threading
import queue
from typing import Callable, Any

from ..core.models import ProcessingOptions
from ..core.processor import CodebaseProcessor


class BackgroundProcessor:
    def __init__(self, message_queue: queue.Queue):
        self.queue = message_queue
        self.is_processing = False

    def start_processing(self, options: ProcessingOptions, 
                        completion_callback: Callable[[Any], None] = None) -> None:
        if self.is_processing:
            return
        
        self.is_processing = True
        thread = threading.Thread(
            target=self._worker_function,
            args=(options, completion_callback),
            daemon=True
        )
        thread.start()

    def _worker_function(self, options: ProcessingOptions, 
                        completion_callback: Callable[[Any], None] = None) -> None:
        try:
            processor = CodebaseProcessor(
                log_callback=lambda msg: self.queue.put(('log', msg)),
                progress_callback=lambda data: self.queue.put(('progress', data))
            )
            
            result = processor.process(options)
            
            if result.files is not None:  # Preview mode
                self.queue.put(('preview_result', {
                    'files': result.files, 
                    'root': options.input_dir
                }))
            
            self.queue.put(('done', result))
            
        except Exception as e:
            import traceback
            self.queue.put(('log', f"!!! AN ERROR OCCURRED: {e}"))
            self.queue.put(('log', traceback.format_exc()))
            self.queue.put(('done', {'success': False, 'message': str(e)}))
        finally:
            self.is_processing = False
            if completion_callback:
                completion_callback(self.is_processing)