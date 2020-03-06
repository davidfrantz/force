/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This file is part of FORCE - Framework for Operational Radiometric 
Correction for Environmental monitoring.

Copyright (C) 2013-2020 David Frantz

FORCE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FORCE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FORCE.  If not, see <http://www.gnu.org/licenses/>.

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/

/**+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
This file contains functions to download files from remote servers 
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/


#include "download-cl.h"

/** libcurl **/
#include <curl.h> // multiprotocol file transfer


typedef struct {
  const char *fname;      // filename
  const char *dname;      // directory name
  char fcat[NPOW_10]; // concatenated filename
  FILE *fp;               // file stream
  long int size;          // file size
} curl_t;


long curl_open_download(struct curl_fileinfo *remote, curl_t *data, int remains);
size_t curl_write_download(void *ptr, size_t size, size_t count, void *data);
long curl_close_download(curl_t *data);


/** This function uses the libcurl library to download a file. The exact 
+++ filename (including path) of the remote file needs to be known. The 
+++ directory, where the local file is to be stored, needs to exist. The 
+++ function returns the curl exit code.
--- f_remote: filename (including path) of remote file
--- f_local:  filename (including path) of local  file (will be created)
--- header:  HTTP header (may be NULL)
+++ Return:   curl exit code
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int download_file(char *f_remote, char *f_local, char *header){
CURL *curl;
CURLcode res;
curl_t data;
struct curl_slist *list = NULL;


  // initialize local data
  data.fname = f_local;
  data.fp = NULL;
 
  // global libcurl initialization
   if ((res = curl_global_init(CURL_GLOBAL_DEFAULT)) != CURLE_OK) return res;
  
  // start easy session
  if(!(curl = curl_easy_init())){
    curl_global_cleanup();
    return CURLE_OUT_OF_MEMORY;
  }

  // set URL
  curl_easy_setopt(curl, CURLOPT_URL, f_remote);
  
  if (header != NULL){
    list = curl_slist_append(list, header);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);
  }

  #ifdef FORCE_DEBUG
  curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
  #endif

  // callback for writing data
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_download);

  // data pointer to pass to the write callback
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);

  // perform file transfer 
  res = curl_easy_perform(curl);

  // end easy handle
  curl_easy_cleanup(curl);

  // close local data
  if(data.fp) fclose(data.fp);

  // global libcurl cleanuo
  curl_global_cleanup();
  
  if (header != NULL) curl_slist_free_all(list);

  return res;
}


/** This function uses the libcurl library to download files that match a
+++ given pattern. The exact filenames do not need to be known, but the 
+++ remote directory needs to be known. Note that pattern argument is a 
+++ concatenation of remote directory and wildcard pattern. All files that
+++ match the given pattern will be downloaded (tiny files < 50KB will be 
+++ skipped). The directory, where the local file is to be stored, needs 
+++ to exist. FTP only. The function returns the curl exit code.
--- d_local: directory, to which files will be downloaded
--- pattern: remote file pattern
--- header:  HTTP header (may be NULL)
+++ Return:  curl exit code
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
int download_pattern(char *d_local, char *pattern, char *header){
CURL *curl;
CURLcode res;
curl_t data;
struct curl_slist *list = NULL;


  // initialize local data
  data.dname = d_local;
  data.fp = NULL;

  // global libcurl initialization
  if ((res = curl_global_init(CURL_GLOBAL_ALL)) != CURLE_OK) return res;
 
  // start easy session
  if(!(curl = curl_easy_init())){
    curl_global_cleanup();
    return CURLE_OUT_OF_MEMORY;
  }
 
  // turn on wildcard matching
  curl_easy_setopt(curl, CURLOPT_WILDCARDMATCH, 1L);
 
  // callback is called before download of concrete file started
  curl_easy_setopt(curl, CURLOPT_CHUNK_BGN_FUNCTION, curl_open_download);
 
  // callback is called after data from the file have been transferred
  curl_easy_setopt(curl, CURLOPT_CHUNK_END_FUNCTION, curl_close_download);
 
  // callback for writing data
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_download);
 
  // put transfer data into callbacks
  curl_easy_setopt(curl, CURLOPT_CHUNK_DATA, &data);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA,  &data);
 
   // set an URL containing wildcard pattern
   curl_easy_setopt(curl, CURLOPT_URL, pattern);
   
  if (header != NULL){
    // set app key
    list = curl_slist_append(list, header);
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, list);
  }

 
  // perform file transfer 
  res = curl_easy_perform(curl);
 
  // end easy handle
  curl_easy_cleanup(curl);

  // global libcurl cleanuo
  curl_global_cleanup();
  
  if (header != NULL) curl_slist_free_all(list);

  return res;
}
 

/** Callback before a transfer with FTP wildcardmatch
+++ This function is called by libcurl before a file is downloaded. Tiny 
+++ files < 50KB will be skipped. The function concatenates the filenames
+++ as retrieved from the remote server via wildcard matching with the 
+++ download directory. The local file is opened then. Note that the call-
+++ back function must match a prototype defined by libcurl (even if some
+++ arguments might be unused).
--- remote:  details about the file, which is to be downloaded
--- data:    local data struct
--- remains: number of chunks remaining per the transfer (unused)
+++ Return:  curl exit code
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
long curl_open_download(struct curl_fileinfo *remote, curl_t *data, int remains){
int nchar;


  if (remote->filetype == CURLFILETYPE_FILE){

    // skip tiny files
    if (remote->size < 50) return CURL_CHUNK_BGN_FUNC_SKIP;

    // concatenate file name
    nchar = snprintf(data->fcat, NPOW_10, "%s/%s", data->dname, remote->filename);
    if (nchar < 0 || nchar >= NPOW_10){ 
      printf("Buffer Overflow in assembling filename\n"); exit(1);}
    
    data->size = remote->size;

    // open local file
    if (!(data->fp = fopen(data->fcat, "w"))){
      return CURL_CHUNK_BGN_FUNC_FAIL;
    }

  }
 
  return CURL_CHUNK_BGN_FUNC_OK;
}


/** Set callback for writing received data 
+++ This function is called by libcurl when downloading files. Received 
+++ data is written to disc. The file is opened if yet closed. The func-
+++ tion returns the number of elements written. Note that the callback 
+++ function must match a prototype defined by libcurl.
--- ptr:    delivered data
--- size:   size in bytes of each element to be written
--- nmemb:  number of elements
--- data:   local data struct
+++ Return: number of written elements
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
size_t curl_write_download(void *ptr, size_t size, size_t count, void *data){
curl_t *out = (curl_t*)data;
size_t written;

  // if file is not open yet
  if (out && !out->fp){

    // open file for writing; return error if not possible
    if (!(out->fp = fopen(out->fname, "wb"))) return -1;

  }

  // write the data
  written = fwrite(ptr, size, count, out->fp);

  return written;
}


/** Callback after a transfer with FTP wildcardmatch
+++ This function is called by libcurl after a file is downloaded. The lo-
+++ cal file is closed. The function compares the size of the downloaded 
+++ and remote files. If sizes do not match, the local file is deleted, an
+++ error code is returned. Note that the callback function must match a 
+++ prototype defined by libcurl.
--- data:    local data struct
+++ Return:  curl exit code
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++**/
long curl_close_download(curl_t *data){
long int size;
char bname[NPOW_10];


  if (data->fp){
  
    // close local data and check how big it is
    fseek(data->fp, 0L, SEEK_END);
    size = ftell(data->fp);
    fclose(data->fp);

    // if size is wrong, try to delete local file
    if (data->size != size){

      basename_with_ext(data->fcat, bname, NPOW_10);
      printf("Download incomplete (%s). Is: %lu. Should: %lu.\n", 
        bname, size, data->size);

      if (remove(data->fcat) == 0){
        printf("Corrupt file deleted.\n");
      } else {
        printf("Failed to delete corrupt file..\n");
      }

      return CURL_CHUNK_END_FUNC_FAIL;
    }

    data->fp = NULL;

  }

  return CURL_CHUNK_END_FUNC_OK;
}
 
