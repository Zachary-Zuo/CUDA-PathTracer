#include "WindowException.h"
#include <sstream>


WindowException::WindowException( int line,const char* file ) noexcept
	:
	line( line ),
	file( file )
{}

const char* WindowException::what() const noexcept
{
	std::ostringstream oss;
	oss << GetType() << std::endl
		<< GetOriginString();
	whatBuffer = oss.str();
	return whatBuffer.c_str();
}

const char* WindowException::GetType() const noexcept
{
	return "Steve Window Exception";
}

int WindowException::GetLine() const noexcept
{
	return line;
}

const std::string& WindowException::GetFile() const noexcept
{
	return file;
}

std::string WindowException::GetOriginString() const noexcept
{
	std::ostringstream oss;
	oss << "[File] " << file << std::endl
		<< "[Line] " << line;
	return oss.str();
}