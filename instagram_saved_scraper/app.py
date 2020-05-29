#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import codecs
import configparser
import errno
import glob
from operator import itemgetter
import json
import logging.config
import hashlib
import os
import pickle
import re
import socket
import sys
import textwrap
import time
import xml.etree.ElementTree as ET
import moviepy.editor as mpe

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

import warnings
import threading
import concurrent.futures
import requests
import requests.packages.urllib3.util.connection as urllib3_connection
import tqdm

from constants import *


class PartialContentException(Exception):
    pass


class InstagramSavedScraper(object):

    def __init__(self, **kwargs):
        default_attr = dict(
            username='',
            login_user=None,
            login_pass=None,
            destination='./',
            logger=None,
            retain_username=False,
            interactive=False,
            quiet=False,
            maximum=0,
            media_metadata=False,
            latest=False,
            latest_stamps=False,
            cookiejar=None,
            verbose=0,
            filter=None,
            proxies={},
            no_check_certificate=False,
            template='{urlname}',
            log_destination='')

        allowed_attr = list(default_attr.keys())
        default_attr.update(kwargs)

        for key in default_attr:
            if key in allowed_attr:
                self.__dict__[key] = default_attr.get(key)

        # Set up a logger
        if self.logger is None:
            self.logger = InstagramSavedScraper.get_logger(level=logging.DEBUG,
                                                           dest=default_attr.get('log_destination'),
                                                           verbose=default_attr.get('verbose'))
        self.posts = []
        self.stories = []

        self.session = requests.Session()
        if self.no_check_certificate:
            self.session.verify = False

        try:
            if self.proxies and type(self.proxies) == str:
                self.session.proxies = json.loads(self.proxies)
        except ValueError:
            self.logger.error("Check is valid json type.")
            raise

        self.session.headers = {'user-agent': CHROME_WIN_UA}
        if self.cookiejar and os.path.exists(self.cookiejar):
            with open(self.cookiejar, 'rb') as f:
                self.session.cookies.update(pickle.load(f))
        self.session.cookies.set('ig_pr', '1')
        self.rhx_gis = ""

        self.cookies = None
        self.authenticated = False
        self.logged_in = False

        if default_attr['filter']:
            self.filter = list(self.filter)
        self.quit = False

    @staticmethod
    def __get_timestamp(item):
        if item:
            for key in ['taken_at_timestamp', 'created_time', 'taken_at', 'date', 'published_time']:
                found = item.get(key, 0)
                try:
                    found = int(found)
                    if found > 1:  # >1 to ignore any boolean casts
                        return found
                except ValueError:
                    pass
        return 0

    @staticmethod
    def get_logger(level=logging.DEBUG, dest='', verbose=0):
        """Returns a logger."""
        logger = logging.getLogger(__name__)

        dest += '/' if (dest != '') and dest[-1] != '/' else ''
        fh = logging.FileHandler(dest + 'instagram-scraper.log', 'w')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        fh.setLevel(level)
        logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        sh_lvls = [logging.ERROR, logging.WARNING, logging.INFO]
        sh.setLevel(sh_lvls[verbose])
        logger.addHandler(sh)

        logger.setLevel(level)

        return logger

    def logout(self):
        """Logs out of instagram."""
        if self.logged_in:
            self.logger.debug('Trying to log out ' + self.login_user)
            try:
                logout_data = {'csrfmiddlewaretoken': self.cookies['csrftoken']}
                self.session.post(LOGOUT_URL, data=logout_data)
                self.authenticated = False
                self.logged_in = False
            except requests.exceptions.RequestException:
                self.logger.warning('Failed to log out ' + self.login_user)

    def authenticate_with_login(self):
        """Logs in to instagram."""
        self.logger.info("Trying to log in: " + self.login_user)
        self.session.headers.update({'Referer': BASE_URL, 'user-agent': STORIES_UA})
        req = self.session.get(BASE_URL)

        self.session.headers.update({'X-CSRFToken': req.cookies['csrftoken']})

        login_data = {'username': self.login_user, 'password': self.login_pass}
        login = self.session.post(LOGIN_URL, data=login_data, allow_redirects=True)
        self.session.headers.update({'X-CSRFToken': login.cookies['csrftoken']})
        self.cookies = login.cookies
        login_text = json.loads(login.text)

        if login_text.get('authenticated') and login.status_code == 200:
            self.authenticated = True
            self.logged_in = True
            self.session.headers.update({'user-agent': CHROME_WIN_UA})
            self.rhx_gis = ""
            self.logger.info('Login successfully for ' + self.login_user)
        else:
            self.logger.error('Login failed for ' + self.login_user)

            if 'checkpoint_url' in login_text:
                checkpoint_url = login_text.get('checkpoint_url')
                self.logger.error('Please verify your account at ' + BASE_URL[0:-1] + checkpoint_url)

                if self.interactive is True:
                    self.login_challenge(checkpoint_url)
            elif 'errors' in login_text:
                for count, error in enumerate(login_text['errors'].get('error')):
                    count += 1
                    self.logger.debug('Session error %(count)s: "%(error)s"' % locals())
            else:
                self.logger.error(json.dumps(login_text))

    def get_dst_dir(self, username):
        """Gets the destination directory and last scraped file time."""
        if self.destination == './':
            dst = './' + username
        else:
            if self.retain_username:
                dst = self.destination + '/' + username
            else:
                dst = self.destination

        return dst

    def get_json(self, *args, **kwargs):
        """Retrieve text from url. Return text as string or None if no data present """
        resp = self.safe_get(*args, **kwargs)

        if resp is not None:
            return resp.text

    def safe_get(self, *args, **kwargs):
        # out of the box solution
        # session.mount('https://', HTTPAdapter(max_retries=...))
        # only covers failed DNS lookups, socket connections and connection timeouts
        # It doesnt work when server terminate connection while response is downloaded
        retry = 0
        retry_delay = RETRY_DELAY
        while True:
            if self.quit:
                return
            try:
                response = self.session.get(timeout=CONNECT_TIMEOUT, cookies=self.cookies, *args, **kwargs)
                if response.status_code == 404:
                    return
                response.raise_for_status()
                content_length = response.headers.get('Content-Length')
                if content_length is not None and len(response.content) != int(content_length):
                    # if content_length is None we repeat anyway to get size and be confident
                    raise PartialContentException('Partial response')
                return response
            except (KeyboardInterrupt):
                raise
            except (requests.exceptions.RequestException, PartialContentException) as e:
                if 'url' in kwargs:
                    url = kwargs['url']
                elif len(args) > 0:
                    url = args[0]
                if retry < MAX_RETRIES:
                    self.logger.warning('Retry after exception {0} on {1}'.format(repr(e), url))
                    self.sleep(retry_delay)
                    retry_delay = min(2 * retry_delay, MAX_RETRY_DELAY)
                    retry = retry + 1
                    continue
                else:
                    keep_trying = self._retry_prompt(url, repr(e))
                    if keep_trying == True:
                        retry = 0
                        continue
                    elif keep_trying == False:
                        return
                raise

    def query_media_gen(self, user, save_dir, end_cursor=''):

        """Generator for media."""
        media, end_cursor, total_count = self.__query_media(user['id'], end_cursor)
        if media:
            try:
                while True:
                    for item in media:
                        item["username"] = user['username']
                        if not self.is_new_media(item, save_dir):
                            return
                        yield item

                    if end_cursor:
                        media, end_cursor, total_count = self.__query_media(user['id'], end_cursor)
                    else:
                        return
            except ValueError:
                self.logger.exception('Failed to query media for user ' + user['username'])

    def is_new_media(self, item, save_dir):
        """Returns True if the media is new."""

        if self.latest is False:
            return True

        exists_all = True

        for full_url, base_name in self.templatefilename(item):
            file_path = os.path.join(save_dir, base_name)
            exists_all = exists_all and os.path.isfile(file_path)

        return not exists_all

    def __query_media(self, id, end_cursor=''):
        self.logger.debug("Query media for {0} with end_cursor {1}".format(id, end_cursor))
        params = SAVED_MEDIA_VARS.format(id, end_cursor)
        self.update_ig_gis_header(params)

        resp = self.get_json(SAVED_MEDIA.format(params))

        if resp is not None:
            payload = json.loads(resp)['data']['user']

            if payload:
                container = payload['edge_saved_media']
                nodes = self._get_nodes(container)
                end_cursor = container['page_info']['end_cursor']
                total_count = container['count']
                self.logger.debug("Query media for {0} result: current_count {1}, total_count {2}, end_cursor {3}".format(id, len(nodes), total_count, end_cursor))
                return nodes, end_cursor, total_count

        return None, None, None

    def get_original_image(self, url):
        """Gets the full-size image from the specified url."""
        # these path parts somehow prevent us from changing the rest of media url
        #url = re.sub(r'/vp/[0-9A-Fa-f]{32}/[0-9A-Fa-f]{8}/', '/', url)
        # remove dimensions to get largest image
        #url = re.sub(r'/[sp]\d{3,}x\d{3,}/', '/', url)
        # get non-square image if one exists
        #url = re.sub(r'/c\d{1,}.\d{1,}.\d{1,}.\d{1,}/', '/', url)
        return url

    def __get_media_details(self, shortcode):
        resp = self.get_json(VIEW_MEDIA_URL.format(shortcode))

        if resp is not None:
            try:
                return json.loads(resp)['graphql']['shortcode_media']
            except ValueError:
                self.logger.warning('Failed to get media details for ' + shortcode)

        else:
            self.logger.warning('Failed to get media details for ' + shortcode)

    def augment_node(self, node):
        details = None

        if 'urls' not in node:
            node['urls'] = []
        if node['is_video'] and 'video_url' in node:
            node['urls'] = [node['video_url']]
        elif '__typename' in node and node['__typename'] == 'GraphImage':
            node['urls'] = [self.get_original_image(node['display_url'])]
        else:
            if details is None:
                details = self.__get_media_details(node['shortcode'])

            if details:
                if '__typename' in details and details['__typename'] == 'GraphVideo':
                    node['urls'] = [details['video_url']]
                elif '__typename' in details and details['__typename'] == 'GraphSidecar':
                    urls = []
                    for carousel_item in details['edge_sidecar_to_children']['edges']:
                        urls += self.augment_node(carousel_item['node'])['urls']
                    node['urls'] = urls
                else:
                    node['urls'] = [self.get_original_image(details['display_url'])]

        return node


    def _get_nodes(self, container):
        return [self.augment_node(node['node']) for node in container['edges']]

    def get_ig_gis(self, rhx_gis, params):
        data = rhx_gis + ":" + params
        if sys.version_info.major >= 3:
            return hashlib.md5(data.encode('utf-8')).hexdigest()
        else:
            return hashlib.md5(data).hexdigest()

    def update_ig_gis_header(self, params):
        self.session.headers.update({
            'x-instagram-gis': self.get_ig_gis(
                self.rhx_gis,
                params
            )
        })

    def worker_wrapper(self, fn, *args, **kwargs):
        try:
            if self.quit:
                return
            return fn(*args, **kwargs)
        except:
            self.logger.debug("Exception in worker thread", exc_info=sys.exc_info())
            raise

    def templatefilename(self, item):

        for url in item['urls']:
            filename, extension = os.path.splitext(os.path.split(url.split('?')[0])[1])
            try:
                template = self.template
                template_values = {
                                    'username' : item['username'],
                                    'owner' : item['owner']['id'],
                                    'urlname': filename,
                                    'shortcode': str(item['shortcode']),
                                    'mediatype' : item['__typename'][5:],
                                   'datetime': time.strftime('%Y%m%d %Hh%Mm%Ss',
                                                             time.localtime(self.__get_timestamp(item))),
                                   'date': time.strftime('%Y%m%d', time.localtime(self.__get_timestamp(item))),
                                   'year': time.strftime('%Y', time.localtime(self.__get_timestamp(item))),
                                   'month': time.strftime('%m', time.localtime(self.__get_timestamp(item))),
                                   'day': time.strftime('%d', time.localtime(self.__get_timestamp(item))),
                                   'h': time.strftime('%Hh', time.localtime(self.__get_timestamp(item))),
                                   'm': time.strftime('%Mm', time.localtime(self.__get_timestamp(item))),
                                   's': time.strftime('%Ss', time.localtime(self.__get_timestamp(item)))}

                customfilename = str(template.format(**template_values) + extension)
                yield url, customfilename
            except KeyError:
                customfilename = str(filename + extension)
                yield url, customfilename

    def _retry_prompt(self, url, exception_message):
        """Show prompt and return True: retry, False: ignore, None: abort"""
        answer = input( 'Repeated error {0}\n(A)bort, (I)gnore, (R)etry or retry (F)orever?'.format(exception_message) )
        if answer:
            answer = answer[0].upper()
            if answer == 'I':
                self.logger.info( 'The user has chosen to ignore {0}'.format(url) )
                return False
            elif answer == 'R':
                return True
            elif answer == 'F':
                self.logger.info( 'The user has chosen to retry forever' )
                global MAX_RETRIES
                MAX_RETRIES = sys.maxsize
                return True
            else:
                self.logger.info( 'The user has chosen to abort' )
                return None

    def make_dir(self, dst):
        try:
            os.makedirs(dst)
        except OSError as err:
            if err.errno == errno.EEXIST and os.path.isdir(dst):
                # Directory already exists
                pass
            else:
                # Target dir exists as a file, or a different error
                raise

    def download(self, item, save_dir='./'):
        """Downloads the media file."""

        files_path = []

        for full_url, base_name in self.templatefilename(item):
            url = full_url.split('?')[0]  # try the static url first, stripping parameters

            file_path = os.path.join(save_dir, base_name)

            if not os.path.exists(os.path.dirname(file_path)):
                self.make_dir(os.path.dirname(file_path))

            if not os.path.isfile(file_path):
                headers = {'Host': urlparse(url).hostname}

                part_file = file_path + '.part'
                downloaded = 0
                total_length = None
                with open(part_file, 'wb') as media_file:
                    try:
                        retry = 0
                        retry_delay = RETRY_DELAY
                        while (True):
                            if self.quit:
                                return
                            try:
                                downloaded_before = downloaded
                                headers['Range'] = 'bytes={0}-'.format(downloaded_before)

                                with self.session.get(url, cookies=self.cookies, headers=headers, stream=True,
                                                      timeout=CONNECT_TIMEOUT) as response:
                                    if response.status_code == 404 or response.status_code == 410:
                                        # on 410 error see issue #343
                                        # instagram don't lie on this
                                        break
                                    if response.status_code == 403 and url != full_url:
                                        # see issue #254
                                        url = full_url
                                        continue
                                    response.raise_for_status()

                                    if response.status_code == 206:
                                        try:
                                            match = re.match(r'bytes (?P<first>\d+)-(?P<last>\d+)/(?P<size>\d+)',
                                                             response.headers['Content-Range'])
                                            range_file_position = int(match.group('first'))
                                            if range_file_position != downloaded_before:
                                                raise Exception()
                                            total_length = int(match.group('size'))
                                            media_file.truncate(total_length)
                                        except:
                                            raise requests.exceptions.InvalidHeader(
                                                'Invalid range response "{0}" for requested "{1}"'.format(
                                                    response.headers.get('Content-Range'), headers.get('Range')))
                                    elif response.status_code == 200:
                                        if downloaded_before != 0:
                                            downloaded_before = 0
                                            downloaded = 0
                                            media_file.seek(0)
                                        content_length = response.headers.get('Content-Length')
                                        if content_length is None:
                                            self.logger.warning(
                                                'No Content-Length in response, the file {0} may be partially downloaded'.format(
                                                    base_name))
                                        else:
                                            total_length = int(content_length)
                                            media_file.truncate(total_length)
                                    else:
                                        raise PartialContentException('Wrong status code {0}', response.status_code)

                                    for chunk in response.iter_content(chunk_size=64 * 1024):
                                        if chunk:
                                            downloaded += len(chunk)
                                            media_file.write(chunk)
                                        if self.quit:
                                            return

                                if downloaded != total_length and total_length is not None:
                                    raise PartialContentException(
                                        'Got first {0} bytes from {1}'.format(downloaded, total_length))

                                break

                            # In case of exception part_file is not removed on purpose,
                            # it is easier to exemine it later when analising logs.
                            # Please do not add os.remove here.
                            except (KeyboardInterrupt):
                                raise
                            except (requests.exceptions.RequestException, PartialContentException) as e:
                                media = url
                                if item['shortcode'] and item['shortcode'] != '':
                                    media += " from https://www.instagram.com/p/" + item['shortcode']
                                if downloaded - downloaded_before > 0:
                                    # if we got some data on this iteration do not count it as a failure
                                    self.logger.warning('Continue after exception {0} on {1}'.format(repr(e), media))
                                    retry = 0  # the next fail will be first in a row with no data
                                    continue
                                if retry < MAX_RETRIES:
                                    self.logger.warning('Retry after exception {0} on {1}'.format(repr(e), media))
                                    self.sleep(retry_delay)
                                    retry_delay = min(2 * retry_delay, MAX_RETRY_DELAY)
                                    retry = retry + 1
                                    continue
                                else:
                                    keep_trying = self._retry_prompt(media, repr(e))
                                    if keep_trying == True:
                                        retry = 0
                                        continue
                                    elif keep_trying == False:
                                        break
                                raise
                    finally:
                        media_file.truncate(downloaded)

                if downloaded == total_length or total_length is None and downloaded > 100:
                    os.rename(part_file, file_path)
                    timestamp = self.__get_timestamp(item)
                    file_time = int(timestamp if timestamp else time.time())
                    os.utime(file_path, (file_time, file_time))

            else:
                self.logger.warning("{0} already exists".format(file_path))

            files_path.append(file_path)

        return files_path

    def sleep(self, secs):
        min_delay = 1
        for _ in range(secs // min_delay):
            time.sleep(min_delay)
            if self.quit:
                return
        time.sleep(secs % min_delay)

    def get_media(self, dst, executor, future_to_item, user):
        """Scrapes the user's posts for media."""

        username = user['username']

        iter = 0
        desc = 'Searching {0} for saved posts'

        for item in tqdm.tqdm(self.query_media_gen(user, dst), desc=desc.format(username),
                              unit=' media', disable=self.quiet):

            if self.is_new_media(item, dst):
                future = executor.submit(self.worker_wrapper, self.download, item, dst)
                future_to_item[future] = item

            if self.media_metadata:
                self.posts.append(item)

            iter = iter + 1
            if self.maximum != 0 and iter >= self.maximum:
                break

    def scrape(self, executor=concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_DOWNLOADS)):
        """Crawls through and downloads user's media"""
        self.session.headers.update({'user-agent': STORIES_UA})
        try:
            self.posts = []
            self.stories = []

            future_to_item = {}

            dst = self.get_dst_dir(self.username)

            # Get the user metadata.
            shared_data = self.get_shared_data(self.username)
            user = self.deep_get(shared_data, 'entry_data.ProfilePage[0].graphql.user')

            if not user:
                self.logger.error(
                    'Error getting user details for {0}. Please verify that the user exists.'.format(self.username))

            self.rhx_gis = ""

            # Crawls the media and sends it to the executor.
            try:

                self.get_media(dst, executor, future_to_item, user)

                # Displays the progress bar of completed downloads. Might not even pop up if all media is downloaded while
                # the above loop finishes.
                if future_to_item:
                    for future in tqdm.tqdm(concurrent.futures.as_completed(future_to_item), total=len(future_to_item),
                                            desc='Downloading', disable=self.quiet):
                        item = future_to_item[future]

                        if future.exception() is not None:
                            self.logger.error(
                                'Media at {0} generated an exception: {1}'.format(item['urls'], future.exception()))

            except ValueError:
                self.logger.error("Unable to scrape user - %s" % self.username)
        finally:
            self.quit = True
            self.logout()

    def get_shared_data(self, username=''):
        """Fetches the user's metadata."""
        resp = self.get_json(BASE_URL + username)

        if resp is not None and '_sharedData' in resp:
            try:
                shared_data = resp.split("window._sharedData = ")[1].split(";</script>")[0]
                return json.loads(shared_data)
            except (TypeError, KeyError, IndexError):
                pass

    def deep_get(self, dict, path):
        def _split_indexes(key):
            split_array_index = re.compile(r'[.\[\]]+')  # ['foo', '0']
            return filter(None, split_array_index.split(key))

        ends_with_index = re.compile(r'\[(.*?)\]$')  # foo[0]

        keylist = path.split('.')

        val = dict

        for key in keylist:
            try:
                if ends_with_index.search(key):
                    for prop in _split_indexes(key):
                        if prop.isdigit():
                            val = val[int(prop)]
                        else:
                            val = val[prop]
                else:
                    val = val[key]
            except (KeyError, IndexError, TypeError):
                return None

        return val

    def save_cookies(self):
        if self.cookiejar:
            with open(self.cookiejar, 'wb') as f:
                pickle.dump(self.session.cookies, f)


def main():
    parser = argparse.ArgumentParser(
        description="instagram-scraper scrapes and downloads an instagram user's photos and videos.",
        epilog=textwrap.dedent("""
        You can hide your credentials from the history, by reading your
        username from a local file:

        $ instagram-scraper @insta_args.txt user_to_scrape

        with insta_args.txt looking like this:
        -u=my_username
        -p=my_password

        You can add all arguments you want to that file, just remember to have
        one argument per line.

        Customize filename:
        by adding option --template or -T
        Default is: {urlname}
        And there are some option:
        {username}: Instagram username(s) to scrape.
        {owner}: Instagram user id(s) to scrape.
        {shortcode}: post shortcode, but profile_pic and story are none.
        {urlname}: filename form url.
        {mediatype}: type of media.
        {datetime}: date and time that photo/video post on,
                     format is: 20180101 01h01m01s
        {date}: date that photo/video post on,
                 format is: 20180101
        {year}: format is: 2018
        {month}: format is: 01-12
        {day}: format is: 01-31
        {h}: hour, format is: 00-23h
        {m}: minute, format is 00-59m
        {s}: second, format is 00-59s

        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars='@')

    parser.add_argument('username', help='Instagram user(s) to scrape', nargs='*')
    parser.add_argument('--destination', '-d', default='./', help='Download destination')
    parser.add_argument('--login-user', '--login_user', '-u', default=None, help='Instagram login user')
    parser.add_argument('--login-pass', '--login_pass', '-p', default=None, help='Instagram login password')
    parser.add_argument('--quiet', '-q', default=False, action='store_true', help='Be quiet while scraping')
    parser.add_argument('--maximum', '-m', type=int, default=0, help='Maximum number of items to scrape')
    parser.add_argument('--retain-username', '--retain_username', '-n', action='store_true', default=False,
                        help='Creates username subdirectory when destination flag is set')
    parser.add_argument('--media-metadata', '--media_metadata', action='store_true', default=False,
                        help='Save media metadata to json file')
    parser.add_argument('--proxies', default={},
                        help='Enable use of proxies, add a valid JSON with http or/and https urls.')
    parser.add_argument('--latest', action='store_true', default=False, help='Scrape new media since the last scrape')
    parser.add_argument('--latest-stamps', '--latest_stamps', default=None,
                        help='Scrape new media since timestamps by user in specified file')
    parser.add_argument('--cookiejar', '--cookierjar', default=None,
                        help='File in which to store cookies so that they can be reused between runs.')
    parser.add_argument('--no-check-certificate', action='store_true', default=False,
                        help='Do not use ssl on transaction')
    parser.add_argument('--interactive', '-i', action='store_true', default=False,
                        help='Enable interactive login challenge solving')
    parser.add_argument('--retry-forever', action='store_true', default=False,
                        help='Retry download attempts endlessly when errors are received')
    parser.add_argument('--verbose', '-v', type=int, default=0, help='Logging verbosity level')
    parser.add_argument('--template', '-T', type=str, default='{urlname}', help='Customize filename template')
    parser.add_argument('--log_destination', '-l', type=str, default='',
                        help='destination folder for the instagram-scraper.log file')
    args = parser.parse_args()

    if args.login_user is None or args.login_pass is None:
        parser.print_help()
        raise ValueError('Must provide login user AND password')

    if not args.username or len(args.username) > 1:
        parser.print_help()
        raise ValueError(
            'Must provide username')

    args.username = args.username[0]

    if args.retry_forever:
        global MAX_RETRIES
        MAX_RETRIES = sys.maxsize

    scraper = InstagramSavedScraper(**vars(args))

    scraper.authenticate_with_login()

    scraper.scrape()

    scraper.save_cookies()


if __name__ == '__main__':
    main()
