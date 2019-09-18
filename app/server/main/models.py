#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# Created Date: Sunday, September 15th 2019, 10:41:07 pm
# Author: Charlene Leong leongchar@myvuw.ac.nz
# Last Modified: Wed Sep 18 2019
###

import os
from werkzeug import secure_filename
from server.__init__ import db


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
    idx = db.Column(db.Integer, primary_key=True, index=True, nullable=False)
    c_label = db.Column(db.Integer, index=True, nullable=False)
    img_grd_idx = db.Column(db.Integer, primary_key=True, index=True,  nullable=True)
    # feat = db.Column(db.Integer, nullable=False)
    img_path = db.Column(db.String, nullable=False)

    def __repr__(self):
        return '<Image {} {} {} {}>\n'.format(self.idx, self.c_label, self.img_grd_idx, self.img_path)
        
    def save_to_db(self):
        db.session.add(self)
        db.session.commit()



def clear_tables():
    meta = db.metadata
    for table in reversed(meta.sorted_tables):
        print ('Clearing {} table...'.format(table))
        db.session.execute(table.delete())
    db.session.commit()
    
# TODO: Refactor task into db 
# https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-xxii-background-jobs
# class Task(db.Model):
#     id = db.Column(db.String(36), primary_key=True)
#     name = db.Column(db.String(128), index=True)
#     description = db.Column(db.String(128))
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
#     complete = db.Column(db.Boolean, default=False)

#     def get_rq_job(self):
#         try:
#             rq_job = rq.job.Job.fetch(self.id, connection=current_app.redis)
#         except (redis.exceptions.RedisError, rq.exceptions.NoSuchJobError):
#             return None
#         return rq_job

#     def get_progress(self):
#         job = self.get_rq_job()
#         return job.meta.get('progress', 0) if job is not None else 100


# if __name__ == '__main__':
#     db.create_all()