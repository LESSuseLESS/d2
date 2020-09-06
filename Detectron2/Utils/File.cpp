#include "Base.h"
#include "File.h"
#include "Utils.h"
#include <windows.h> 

using namespace std;
using namespace Detectron2;

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// file path functions

std::string File::GetCwd() {
	TCHAR buf[MAX_PATH];
	auto ret = GetCurrentDirectory(MAX_PATH, buf);
	verify(ret > 0);
	return buf;
}

void File::SetCwd(const std::string &cwd) {
	::SetCurrentDirectory(cwd.c_str());
}

bool File::IsAbsolutePath(const std::string &pathname) {
	return pathname.find(":") == 1; // ouch: for Windows only
}

bool File::IsDir(const std::string &pathname) {
	struct stat s;
	return stat(pathname.c_str(), &s) == 0 && (s.st_mode & S_IFDIR);
}

bool File::IsFile(const std::string &pathname) {
	struct stat s;
	return stat(pathname.c_str(), &s) == 0 && (s.st_mode & S_IFREG);
}

std::string File::Dirname(const std::string &pathname) {
	auto pos = pathname.rfind('/');
	if (pos == string::npos) {
		pos = pathname.rfind('\\');
	}
	if (pos == string::npos) {
		return "";
	}
	return pathname.substr(0, pos);
}

std::string File::Basename(const std::string &pathname) {
	auto pos = pathname.rfind('/');
	if (pos == string::npos) {
		pos = pathname.rfind('\\');
	}
	if (pos == string::npos) {
		return pathname;
	}
	return pathname.substr(pos + 1);
}

std::string File::ComposeFilename(const std::string &dirname, const std::string &basename) {
	auto last = dirname[dirname.size()];
	if (last == '/' || last == '\\') {
		return dirname + basename;
	}
	return dirname + '\\' + basename;
}

std::string File::ReplaceExtension(const std::string &pathname, const std::string &new_extension) {
	auto filename = Basename(pathname);
	auto pos = filename.rfind('.');
	if (pos != string::npos) {
		return pathname.substr(0, pathname.size() - (filename.size() - pos) + 1) + new_extension;
	}
	return pathname + '.' + new_extension;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

File::File(const string &fullpath, bool read) : m_filename(fullpath), m_file(nullptr) {
	errno_t err = fopen_s(&m_file, m_filename.c_str(), read ? "rb" : "w");
	verify(err == 0);
	verify(m_file != nullptr);
	int res = fseek(m_file, 0, SEEK_SET);
	Verify(res == 0);
}

void File::Seek(int offset) {
	int res = fseek(m_file, offset, SEEK_SET);
	Verify(res == 0);
}

int File::ReadInt() {
	char buf[4];
	size_t len = fread(buf, 1, 4, m_file);
	Verify(len == 4);
	swap(buf[0], buf[3]);
	swap(buf[1], buf[2]);
	int ret;
	memcpy(&ret, buf, 4);
	return ret;
}

void File::Read(char *buf, size_t total) {
	while (total > 0) {
		size_t len = fread(buf, 1, total, m_file);
		Verify(len > 0 && len <= total);
		total -= len;
		buf += len;
	}
}

std::string File::Read() {
	fseek(m_file, 0, SEEK_END);
	long fsize = ftell(m_file);
	fseek(m_file, 0, SEEK_SET);

	char *buf = (char *)malloc(fsize + 1);
	fread(buf, 1, fsize, m_file);
	string ret = buf; // ouch
	free(buf);
	return ret;
}

void File::Write(const std::string &content) {
	Write(content.c_str(), content.length());
}

void File::Write(const char *buf, size_t total) {
	while (total > 0) {
		size_t len = fwrite(buf, 1, total, m_file);
		Verify(len > 0 && len <= total);
		total -= len;
		buf += len;
	}
}

void File::Close() {
	fclose(m_file);
}

void File::Verify(bool expr) {
	if (!expr) {
		if (ferror(m_file)) {
			perror("Error: ");
		}
		else {
			cerr << "ferror() was false";
		}
		verify(false);
	}
}
