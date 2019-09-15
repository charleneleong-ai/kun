#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 10:41:07 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Sun Sep 15 2019
###

import os
from werkzeug import secure_filename
from server import db

class FilesystemObject(object):
    def __init__(self, filename, post=None, root=None):
        """Create an object from the information of the given filename or from a
        uploaded file.

        Example of usage:

            if request.method == 'POST' and 'photo' in request.POST:
                f = FilesystemObject('cats.png', request.POST['photo'])

        """
        self.root_dir = root
        self.filename = filename if not post else secure_filename(post.filename)
        self.abspath  = os.path.join(self.root_dir, filename)

        if post:
            self.upload(post)

        try:
            stats = os.stat(self.abspath)
        except IOError as e:
            raise FilesystemObjectDoesNotExist(e.message)

        self.timestamp = stats.st_mtime

    def upload(self, post):
        """Get a POST file and save it to the settings.GALLERY_ROOT_DIR"""
        # TODO: handle filename conflicts
        # http://flask.pocoo.org/docs/patterns/fileuploads/
        post.save(os.path.join(self.root_dir, self.filename))

    @classmethod
    def all(cls, root):
        """Return a list of files contained in the directory pointed by settings.GALLERY_ROOT_DIR.
        """
        return [cls(x) for x in os.listdir(root)]

class FilesystemObjectDoesNotExist(Exception):
    pass

# class Image(FilesystemObject):
#     pass

class Image(db.Model):
    idx = db.Column(db.Integer, primary_key=True, nullable=False)
    label = db.Column(db.Integer, nullable=False)
    image_path = db.Column(db.String, nullable=False)

    def __repr__(self):
        return '<Image {}_{}_{}>'.format(idx, label, image_path)