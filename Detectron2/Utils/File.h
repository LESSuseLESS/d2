#pragma once

#include <Detectron2/Base.h>

namespace Detectron2
{
	class File {
	public:
		static std::string GetCwd();
		static void SetCwd(const std::string &cwd);

		static bool IsAbsolutePath(const std::string &pathname);
		static bool IsDir(const std::string &pathname);
		static bool IsFile(const std::string &pathname);

		static std::string Dirname(const std::string &pathname);
		static std::string Basename(const std::string &pathname);
		static std::string ComposeFilename(const std::string &dirname, const std::string &basename);
		static std::string ReplaceExtension(const std::string &pathname, const std::string &new_extension);

	public:
		File(const std::string &fullpath, bool read = true);
		void Close();

		std::string Read();
		void Write(const std::string &content);

		void Seek(int offset);
		int ReadInt();
		void Read(char *buf, size_t total);
		void Write(const char *buf, size_t total);

	private:
		std::string m_filename;
		FILE *m_file;

		void Verify(bool expr);
	};
}
