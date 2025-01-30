import sys
import traceback

def error_handling(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error: {str(error)}\n"
        f"Line number: {str(exc_tb.tb_lineno)}\n"
        f"Error message: {str(error_detail)}\n"
        f"In file: {str(file_name)}"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Get the detailed error message
        detailed_error_message = error_handling(error_message, error_detail)
        
        # Initialize the base Exception class with the detailed error message
        super().__init__(detailed_error_message)
        
        # Store the detailed error message in an instance variable
        self.error_message = detailed_error_message

# if __name__ == "__main__":
#     try:
#         raise CustomException("An error occurred", sys.exc_info())
#     except CustomException as e:
#         print(e.error_message)
#         print("Custom exception caught")
#     except Exception as e:
#         print("Exception caught")
#         print(e)